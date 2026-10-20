"""Cropr-style token pruning for the Report 2 BaselineModel interface.

Wrap an already constructed model with CroprModel. Train with main task loss
PLUS wrapper.auxiliary_loss(target); hard top-k alone cannot train the scorer.
This is a block-level adapter, not an attention mask or pretrained pruner.
See CROPR.md for usage, supported architectures and limitations.
"""
import os
from pathlib import Path
import types
import hashlib
from contextlib import ExitStack
import math

import torch
from torch import nn
from torch.nn import functional as F


def _gather(x, indices):
    return x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def make_schedule(depth, pruning_rate=8, *, mode="gradual", fusion=True):
    """Return {zero-based block index: number of patch tokens to remove}.

    Three-stage mode distributes the same total removal over three stages.
    With fusion, do not prune after the penultimate or last block.
    """
    if type(pruning_rate) is not int or pruning_rate < 0:
        raise ValueError("pruning_rate must be a nonnegative integer count")
    stages = depth - (2 if fusion else 1)
    if stages < 1:
        raise ValueError("Backbone is too shallow for this schedule")
    if mode == "gradual":
        return dict.fromkeys(range(stages), pruning_rate)
    if mode != "three_stage" or stages < 3:
        raise ValueError("Use gradual, or three_stage with at least three eligible blocks")
    locations = [0, (stages - 1) // 2, stages - 1]
    quotient, remainder = divmod(stages * pruning_rate, 3)
    return {i: quotient + (j < remainder) for j, i in enumerate(locations)}


class Cropr(nn.Module):
    """Learned query/key scores with a detached auxiliary prediction branch.

    Default identity Q/K/V projections and one head match the inexpensive
    source configuration. Multiple queries support a fixed segmentation grid.
    """
    def __init__(self, embed_dim, num_classes, num_queries=1, num_heads=1):
        super().__init__()
        if min(embed_dim, num_classes, num_queries, num_heads) < 1 or embed_dim % num_heads:
            raise ValueError("Positive dimensions required; heads must divide embed_dim")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # nn.parameter makes this a learnable parameter
        # cls uses one query
        # seg uses one query per original patch grid position
        self.queries = nn.Parameter(torch.empty(1, num_queries, embed_dim))  
        nn.init.trunc_normal_(self.queries, std=embed_dim ** -0.5)
        self.proj = nn.Linear(embed_dim, embed_dim) if num_heads > 1 else nn.Identity()
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # nn.sequential is just linear layer 1, gelu, LL2
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
                                 nn.Linear(4 * embed_dim, embed_dim))
        # layer norm normalizes feature values within each token vector.  normalized = (x-mean) / sqrt(var + epsilon) ---- out = learned_scale * normalized + learned_bias
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
    
    #forward takes the vit attn block output sequence as input using a
    def forward(self, x, *, auxiliary=False):
        if not auxiliary:
            # sum_{heads,queries} Q K^T = X @ sum_queries(Q), for identity Q/K.
            # No softmax/value aggregation/MLP/head at inference.
            return (x @ self.queries.sum(dim=1).transpose(0, 1)).squeeze(-1), None
        tokens = x.detach()  # auxiliary loss must not update the backbone
        batch, count, dim = tokens.shape
        q = self.queries.expand(batch, -1, -1).reshape(
            batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = tokens.reshape(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        # attn operation
        attn = q @ kv.transpose(-2, -1)
        # softmax
        weights = (attn * self.head_dim ** -0.5).softmax(-1)

        aggregate = (weights @ kv)
        aggregate = aggregate.transpose(1, 2).reshape(batch, -1, dim)
        aggregate = self.proj(aggregate)
        # last two lines are the auxiliary head, which is used for training only. The auxiliary head is a linear layer that maps the aggregated features to the number of classes. The output of the auxiliary head is used to compute the auxiliary loss during training.
        aggregate = aggregate + self.mlp(self.mlp_norm(aggregate))  

        # attn.sum((1,2)) is the token scores that are used to select top-k toekns to keep
        # self.head(aggregate) is the aux head that is used to calculate the aux training loss
        # loss then back props through the model and gives training signal for scorer that determines top-k selection   
        return attn.sum((1, 2)), self.head(aggregate)


def _architecture(backbone):
    name, vision = backbone.name, backbone.model
    if name == "dinov2":
        if getattr(vision, "chunked_blocks", False):
            raise ValueError("Chunked DINO blocks are unsupported; construct with block_chunks=0")
        return vision.blocks, 1 + getattr(vision, "num_register_tokens", 0)
    if name in {"siglip", "siglip2"}:
        if "Siglip2" in type(vision).__name__:
            raise ValueError("NaFlex Siglip2 padding/spatial masks require a separate adapter")
        return vision.encoder.layers, 0
    if name == "radio":
        inner = vision.model
        generator = getattr(inner, "patch_generator", None)
        if generator is None or not hasattr(inner, "blocks"):
            raise ValueError("Only RADIO CPE ViT backbones are supported")
        if getattr(vision, "adaptors", None):
            raise ValueError("Construct RADIO with adaptor_names=None")
        return inner.blocks, generator.num_skip
    raise ValueError(f"Unsupported backbone: {name}")


class CroprModel(nn.Module):
    """Wrap Report 2 BaselineModel, preserving its logits and dense head interface.

    Hooks are scoped to each forward and always removed, including on errors.
    Freeze the backbone by default; auxiliary scorers and the task head train.
    Use one wrapper per task. Eager, single-forward execution only (no concurrent
    forwards, checkpoint recomputation, torch.compile or intermediate outputs).
    """
    def __init__(self, model, *, task="classification", num_classes=None,
                 pruning_rate=8, schedule=None, schedule_mode="gradual",
                 fusion=True, grid_size=(32, 32), num_heads=1, freeze_backbone=True):
        super().__init__()
        if task not in {"classification", "segmentation"}:
            raise ValueError("task must be classification or segmentation")
        if task == "segmentation" and not fusion:
            raise ValueError("Dense segmentation requires last-layer fusion")
        blocks, prefix = _architecture(model.backbone)
        if any(getattr(m, "gradient_checkpointing", False) or
               getattr(m, "grad_checkpointing", False) for m in model.modules()):
            raise ValueError("Disable gradient checkpointing before wrapping")
        if any("vitdet" in type(m).__module__.lower() for m in model.modules()):
            raise ValueError("Windowed RADIO/VitDet is unsupported")
        if hasattr(model.backbone.forward, "__wrapped__"):
            raise ValueError("Remove the backbone's no_grad decorator before training Cropr")
        self.model = model
        self.task, self.fusion, self.prefix = task, fusion, prefix
        self.grid_size = tuple(grid_size)
        if len(self.grid_size) != 2 or any(type(n) is not int or n < 1 for n in self.grid_size):
            raise ValueError("grid_size must contain two positive integers")
        self.schedule = (make_schedule(len(blocks), pruning_rate, mode=schedule_mode, fusion=fusion)
                         if schedule is None else dict(schedule))
        limit = len(blocks) - (2 if fusion else 1)
        if not self.schedule or any(type(i) is not int or not 0 <= i < limit or
                                   type(n) is not int or n < 0 for i, n in self.schedule.items()):
            raise ValueError("Invalid pruning schedule: use eligible block indices and nonnegative counts")
        classes = num_classes if num_classes is not None else (1000 if task == "classification" else 150)
        queries = 1 if task == "classification" else math.prod(self.grid_size)
        self.pruners = nn.ModuleDict({str(i): Cropr(model.backbone.feature_dim, classes, queries, num_heads)
                                     for i in self.schedule})
        reference = next(model.backbone.parameters())
        self.pruners.to(device=reference.device, dtype=reference.dtype)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            model.backbone.requires_grad_(False)
        self.auxiliary_predictions = []
        self.token_stats = []
        self._active = False
        self.train(model.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.model.backbone.eval()
        return self

    def forward(self, images, task=None, **kwargs):
        if task is not None and task != self.task:
            raise ValueError("This wrapper is configured for a different task")
        if self._active:
            raise RuntimeError("Concurrent/reentrant Cropr forward is unsupported")
        if self.training and not torch.is_grad_enabled():
            raise RuntimeError("Cropr training needs gradients; use eval() for inference")
        blocks, _ = _architecture(self.model.backbone)
        auxiliary = self.training
        self.auxiliary_predictions, self.token_stats = [], []
        ids, removed, removed_ids = None, [], []

        def before(index, module, args, kw):
            nonlocal ids
            x = args[0] if args else kw.get("hidden_states")
            if not isinstance(x, torch.Tensor) or x.ndim != 3:
                raise ValueError("Expected B,N,D transformer tokens")
            if any(kw.get(key) is not None for key in ("attention_mask", "attn_mask", "rope")):
                raise ValueError("Masked/rotary blocks need position-aware pruning support")
            if index == 0:
                patches = x.shape[1] - self.prefix
                if sum(self.schedule.values()) >= patches:
                    raise ValueError("Schedule must retain at least one patch token")
                if self.task == "segmentation" and patches != math.prod(self.grid_size):
                    raise ValueError("grid_size must match the input patch grid")
                ids = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
            if index == len(blocks) - 1 and self.fusion and removed:
                x = _gather(torch.cat([x] + removed, 1), torch.cat([ids] + removed_ids, 1).argsort(1))
                if args:
                    args = (x,) + args[1:]
                else:
                    kw = dict(kw, hidden_states=x)
            self.token_stats.append({"block": index, "input_tokens": x.shape[1],
                                     "removed": self.schedule.get(index, 0)})
            return args, kw

        def after(index, module, args, output):
            nonlocal ids
            if index not in self.schedule:
                return output
            x = output[0] if isinstance(output, tuple) else output
            if not isinstance(x, torch.Tensor) or x.ndim != 3:
                raise ValueError("Unsupported transformer block output")
            scores, pred = self.pruners[str(index)](x, auxiliary=auxiliary)
            if pred is not None:
                self.auxiliary_predictions.append(pred)
            count = self.schedule[index]
            if not count:
                return output
            # Keep all prefix/register tokens in their original slots. Maintain
            # spatial order among selected patches; rank only patch tokens.

            # This keeps the highest-scoring patches and drops the lowest-scoring count patches.
            # It implements top-k selection through sorting and slicing
            rank = scores[:, self.prefix:].argsort(dim=1, descending=True)
            keep_patch, drop = rank[:, :-count].sort(1).values + self.prefix, rank[:, -count:] + self.prefix
            prefix = torch.arange(self.prefix, device=x.device).expand(x.shape[0], -1)
            keep = torch.cat([prefix, keep_patch], 1)
            if self.fusion:
                removed.append(_gather(x, drop))
                removed_ids.append(ids.gather(1, drop))
            ids = ids.gather(1, keep)
            x = _gather(x, keep)
            return (x,) + output[1:] if isinstance(output, tuple) else x

        self._active = True
        try:
            with ExitStack() as stack:
                for i, block in enumerate(blocks):
                    handle = block.register_forward_pre_hook(
                        lambda m, a, k, i=i: before(i, m, a, k), with_kwargs=True)
                    stack.callback(handle.remove)
                    handle = block.register_forward_hook(lambda m, a, o, i=i: after(i, m, a, o))
                    stack.callback(handle.remove)
                return self.model(images, self.task, **kwargs)
        except Exception:
            self.auxiliary_predictions = []
            raise
        finally:
            self._active = False

    def auxiliary_loss(self, target, *, ignore_index=255):
        """Sum per-pruner CE losses as in the source; call after training forward.

        Segmentation queries correspond to the original grid, never the selected
        token order. Labels are nearest-resized to that grid. Hard labels only.
        """
        if not self.auxiliary_predictions:
            raise RuntimeError("Run a training forward before requesting auxiliary loss")
        losses = []
        for prediction in self.auxiliary_predictions:
            if self.task == "classification":
                logits, labels = prediction[:, 0].float(), target
            else:
                logits = prediction.transpose(1, 2).reshape(
                    prediction.shape[0], prediction.shape[-1], *self.grid_size).float()
                labels = F.interpolate(target[:, None].float(), size=self.grid_size,
                                       mode="nearest")[:, 0].long()
            valid = labels != ignore_index
            losses.append(F.cross_entropy(logits, labels, ignore_index=ignore_index,
                                           reduction="sum") / valid.sum().clamp_min(1))
        return torch.stack(losses).sum()

    @property
    def backbone(self):
        return self.model.backbone


    @property
    def classification_head(self):
        return self.model.classification_head


    @property
    def segmentation_head(self):
        return self.model.segmentation_head


    def auxiliary_loss_sums(self, target, *, ignore_index=255):
        """Return summed CE across stages and valid target count for accumulation."""
        if not self.auxiliary_predictions:
            raise RuntimeError("Run a training forward before requesting auxiliary loss")
        if self.task == "classification":
            labels = target
        else:
            labels = F.interpolate(target[:, None].float(), size=self.grid_size,
                                   mode="nearest")[:, 0].long()
        losses = []
        for prediction in self.auxiliary_predictions:
            logits = (prediction[:, 0] if self.task == "classification" else
                      prediction.transpose(1, 2).reshape(
                          prediction.shape[0], prediction.shape[-1], *self.grid_size))
            losses.append(F.cross_entropy(logits.float(), labels, ignore_index=ignore_index,
                                          reduction="sum"))
        return torch.stack(losses).sum(), int((labels != ignore_index).sum().item())



def resolve_project_root(root=None):
    """Locate existing datasets/models without downloading or writing outside Report 2."""
    if root is not None:
        return Path(root).expanduser().resolve()
    for candidate in (Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parents[1]):
        if (candidate / "models").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError("Pass ROOT pointing to the project containing models/ and datasets/")


class FrozenBackbone(nn.Module):
    def __init__(self, name, *, root=None):
        super().__init__()
        self.name = name
        root = resolve_project_root(root)
        SOURCE = root / "models/source"
        CHECKPOINTS = root / "models/checkpoints"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["XFORMERS_DISABLED"] = "1"

        if name == "dinov2":
            self.model = torch.hub.load(
                str(SOURCE / "dinov2"),
                "dinov2_vitb14",
                source="local",
                pretrained=False,
            )

            weights = torch.load(
                CHECKPOINTS / "dinov2" / "dinov2_vitb14_pretrain.pth",
                map_location="cpu",
                weights_only=True,
            )
            self.model.load_state_dict(weights, strict=True)

            self.feature_dim = self.model.embed_dim
            self.patch_size = 14
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        elif name == "siglip2":
            from transformers import AutoConfig, AutoModel
            from safetensors.torch import load_file
            folder = CHECKPOINTS / "siglip2"

            config = AutoConfig.from_pretrained(
                str(folder),
                local_files_only=True,
            )

            full_model = AutoModel.from_config(config)
            weights = load_file(
                str(folder / "model.safetensors"),
                device="cpu",
            )
            full_model.load_state_dict(weights, strict=True)

            self.model = full_model.vision_model
            self.feature_dim = config.vision_config.hidden_size
            self.patch_size = config.vision_config.patch_size

            del full_model, weights

            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)

        elif name == "radio":
            checkpoint = (
                CHECKPOINTS / "radio" / "radio-v2.5-b_half.pth.tar"
            ).resolve()

            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

            self.model = torch.hub.load(
                str(SOURCE / "radio"),
                "radio_model",
                source="local",
                version=str(checkpoint),
                adaptor_names=None,
            )

            self.feature_dim = self.model.embed_dim
            self.patch_size = 16

            # RADIO performs its own normalization.
            mean = (0.0, 0.0, 0.0)
            std = (1.0, 1.0, 1.0)

        else:
            raise ValueError(f"Unknown model: {name}")

        self.model.float().requires_grad_(False).eval()

        if name == "radio":
            self.model.input_conditioner.dtype = torch.float32

        self.register_buffer(
            "mean", torch.tensor(mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std).view(1, 3, 1, 1)
        )

    def train(self, mode=True):
        # Preserve Report 1's backbone evaluation behavior.
        super().train(False)
        return self

    # Autograd remains enabled for the Cropr auxiliary branches.
    def forward(self, images):
        if images.shape[-2:] != (448, 448):
            raise ValueError("Expected 448 × 448 images.")

        images = (images - self.mean) / self.std

        if self.name == "dinov2":
            patches = self.model.forward_features(
                images
            )["x_norm_patchtokens"]

        elif self.name == "siglip2":
            patches = self.model(
                pixel_values=images,
                interpolate_pos_encoding=True,
                return_dict=True,
            ).last_hidden_state

        else:
            _, patches = self.model(images, feature_fmt="NLC")

        grid = (
            images.shape[-2] // self.patch_size,
            images.shape[-1] // self.patch_size,
        )
        return patches.float(), grid

class ClassificationHead(nn.Module):
    def __init__(self, feature_dim, num_classes=1000):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, patches):
        return self.classifier(patches.mean(dim=1))


class SegmentationHead(nn.Module):
    def __init__(self, feature_dim, num_classes=150, width=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, width, 1, bias=False),
            nn.GroupNorm(32, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(32, width),
            nn.GELU(),
            nn.Conv2d(width, num_classes, 1),
        )

    def forward(self, patches, grid_size, output_size):
        batch, count, channels = patches.shape
        height, width = grid_size

        if count != height * width:
            raise ValueError("Tokens do not match the spatial grid.")

        features = patches.transpose(1, 2).reshape(
            batch, channels, height, width
        )

        return F.interpolate(
            self.decoder(features),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class BaselineModel(nn.Module):
    def __init__(self, name, *, root=None):
        super().__init__()
        self.backbone = FrozenBackbone(name, root=root)
        dim = self.backbone.feature_dim
        self.classification_head = ClassificationHead(dim)
        self.segmentation_head = SegmentationHead(dim)

    def forward(self, images, task, output_size=None):
        patches, grid = self.backbone(images)

        if task == "classification":
            return self.classification_head(patches)

        if task == "segmentation":
            if output_size is None:
                output_size = images.shape[-2:]

            return self.segmentation_head(
                patches, grid, output_size
            )

        raise ValueError(f"Unknown task: {task}")


VARIANTS = {
    "gradual_llf": ("gradual", True),
    "gradual_no_llf": ("gradual", False),
    "three_stage_llf": ("three_stage", True),
}


def build_cropr_model(name, task, *, root=None, variant="gradual_llf",
                      pruning_rate=8, input_size=448, num_heads=1, base=None):
    """Construct one complete experiment, without optimizer or training side effects.

    Optional base is useful for integration tests; normal runs load local weights.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    mode, fusion = VARIANTS[variant]
    if task not in {"classification", "segmentation"}:
        raise ValueError(f"Unknown task: {task}")
    if task == "segmentation" and not fusion:
        raise ValueError("Segmentation requires LLF to restore its dense grid")
    if input_size != 448 and base is None:
        raise ValueError("Report 1's pretrained model interface expects input_size=448")
    base = BaselineModel(name, root=root) if base is None else base
    inactive = "segmentation" if task == "classification" else "classification"
    setattr(base, f"{inactive}_head", nn.Identity())
    patch_size = base.backbone.patch_size
    if input_size % patch_size:
        raise ValueError("Input size must be divisible by the patch size")
    grid = (input_size // patch_size,) * 2
    model = CroprModel(base, task=task, num_classes=1000 if task == "classification" else 150,
                       pruning_rate=pruning_rate, schedule_mode=mode, fusion=fusion,
                       grid_size=grid, num_heads=num_heads)
    if sum(model.schedule.values()) >= math.prod(grid):
        raise ValueError("Schedule must retain at least one patch token")
    return model


def load_cropr_checkpoint(checkpoint, *, root=None, device="cpu"):
    """Rebuild from local pretrained weights and restore the learned head/pruners.

    Checkpoints are for evaluation/reconstruction, not optimizer-state resume.
    """
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("training_mode") != "cropr":
        raise ValueError("Expected a Cropr checkpoint")
    try:
        from .report2_cropr_runner import _state_digest
    except ImportError:
        from report2_cropr_runner import _state_digest
    spec = checkpoint["cropr"]
    model = build_cropr_model(checkpoint["model_name"], checkpoint["task"], root=root, **spec)
    if _state_digest(model.backbone.state_dict()) != checkpoint["pretrained_backbone_sha256"]:
        raise ValueError("Local pretrained backbone differs from the training checkpoint")
    model.pruners.load_state_dict(checkpoint["pruner_state_dict"], strict=True)
    getattr(model, f"{model.task}_head").load_state_dict(checkpoint["head_state_dict"], strict=True)
    return model.to(device).eval()


class LoRALinear(nn.Module):
    """Frozen linear plus (alpha/rank) * B(Ax).  Zero B preserves initial logits.

    Fused QKV gets one rank-r update over the whole fused weight, not three
    independent rank-r updates.  The exact targets and parameter count are saved.
    """
    def __init__(self, base, rank, alpha):
        super().__init__()
        if type(base) is not nn.Linear:
            raise TypeError("LoRA requires an ordinary nn.Linear; review custom projections explicitly.")
        if rank > min(base.in_features, base.out_features):
            raise ValueError("LoRA rank exceeds a projection dimension.")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        # Preserve correctness for attention implementations using F.linear
        # directly instead of calling their projection module's forward.
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling



def _lora_modules(backbone):
    return {n: m for n, m in backbone.named_modules() if isinstance(m, LoRALinear)}



def _state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()



def _frozen_backbone_digest(backbone):
    # Normalize adapter wrapper keys back to their original pretrained names.
    prefixes = tuple(_lora_modules(backbone))
    state = {}
    for name, value in backbone.state_dict().items():
        if any(name == f"{p}.lora_A" or name == f"{p}.lora_B" for p in prefixes):
            continue
        for p in prefixes:
            if name.startswith(p + ".base."):
                name = p + "." + name[len(p + ".base."):]
                break
        state[name] = value
    return _state_digest(state)



def inject_lora(model, *, rank=8, alpha=16.0, targets=None,
                unwrap_backbone_forward=True):
    """Inject attention adapters in an existing model instance only.

    targets, if supplied, are exact Linear paths relative to model.backbone.
    Otherwise recognize fused qkv/proj or complete q_proj/k_proj/v_proj attention.
    Never adapt pooling-only attention or MLPs automatically.
    """
    backbone = model.backbone
    if _lora_modules(backbone):
        raise ValueError("Backbone already contains LoRA adapters.")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer.")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be positive and finite.")
    modules = dict(backbone.named_modules())
    if targets is None:
        targets = []
        for name, module in modules.items():
            if name.rsplit(".", 1)[-1] not in {"attn", "self_attn", "attention"}:
                continue
            children = dict(module.named_children())
            if "qkv" in children:
                leaves = ["qkv"] + (["proj"] if "proj" in children else [])
            elif {"q_proj", "k_proj", "v_proj"}.issubset(children):
                leaves = ["q_proj", "k_proj", "v_proj"]
                leaves += [k for k in ("out_proj", "o_proj") if k in children]
            else:
                continue
            targets.extend(f"{name}.{leaf}" for leaf in leaves)
    targets = list(targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("No attention projections found, or duplicate LoRA targets.  Supply exact LORA_TARGETS.")
    for name in targets:
        if name not in modules or type(modules[name]) is not nn.Linear:
            raise ValueError(f"Unsupported LoRA target: {name}")
        if rank > min(modules[name].in_features, modules[name].out_features):
            raise ValueError(f"LoRA rank exceeds dimensions of {name}")
    backbone.requires_grad_(False)
    for name in targets:
        parent, _, leaf = name.rpartition(".")
        setattr(backbone.get_submodule(parent), leaf, LoRALinear(modules[name], rank, alpha))
    unwrapped = False
    forward = backbone.forward
    # The supplied FrozenBackbone has a single @torch.no_grad decorator.
    # Remove one wrapper on this instance; never mutate the notebook class.
    if unwrap_backbone_forward and hasattr(forward, "__wrapped__"):
        backbone.forward = types.MethodType(forward.__wrapped__, backbone)
        unwrapped = True
    backbone.eval()
    return {"rank": rank, "alpha": float(alpha), "dropout": 0.0,
            "targets": targets, "unwrapped_backbone_forward": unwrapped,
            "fused_qkv_policy": "one rank-r update over fused QKV weight",
            "trainable_parameters": sum(p.numel() for p in backbone.parameters() if p.requires_grad)}



def _adapter_state(backbone):
    return {f"{n}.{key}": getattr(m, key).detach().cpu().clone()
            for n, m in _lora_modules(backbone).items() for key in ("lora_A", "lora_B")}



def _load_adapter_state(backbone, state):
    parameters = {f"{n}.{key}": getattr(m, key)
                  for n, m in _lora_modules(backbone).items() for key in ("lora_A", "lora_B")}
    if parameters.keys() != state.keys():
        raise ValueError("Adapter checkpoint target keys differ from this model.")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape or not torch.isfinite(state[name]).all():
                raise ValueError(f"Invalid adapter checkpoint tensor: {name}")
            parameter.copy_(state[name])



@torch.no_grad()
def merge_lora(model):
    """Merge once and remove adapters, leaving ordinary frozen Linear modules."""
    modules = _lora_modules(model.backbone)
    if not modules:
        raise ValueError("No unmerged adapters found.")
    for name, module in modules.items():
        merged = module.weight
        if not torch.isfinite(merged).all():
            raise RuntimeError(f"Non-finite merged weight: {name}")
        module.base.weight.copy_(merged)
        parent, _, leaf = name.rpartition(".")
        setattr(model.backbone.get_submodule(parent), leaf, module.base)
    model.backbone.requires_grad_(False).eval()

