"""Frozen cached-head and optional uncached LoRA baseline experiments.

TRAIN_PRECISION="bf16" uses BF16 autocast for training forwards, with FP32
parameters and loss. EVAL_TF32 optionally enables TF32 only in evaluate().
COMPILE_LORA optionally compiles the training forward, with eager retry on failure.
Merge audits, cache extraction and inference benchmarks keep TF32 disabled.
LOADER_BACKEND="thread", NUM_WORKERS=2 enables bounded CPU image prefetch in two
threads without Windows child interpreters. "process" retains spawn DataLoaders.
FAIL_FAST=True saves errors and stops instead of allocating another model.


TRAINING_MODE="lora" is the default: attention adapters plus the task head
train from RGB, never cached features.  Optional cached_head mode trains only heads.
LoRA defaults: rank 8, alpha 16, LR 1e-4, dropout 0; separate head LR.
The runner handles the supplied FrozenBackbone's @torch.no_grad wrapper on
its own model instance. Nested inference/no_grad blocks fail the gradient audit.
Best/last LoRA checkpoints contain adapters AND head, plus a base-weight hash.
Use load_baseline_checkpoint on the same original pretrained model to reload.
No automatic resume. Inference measurements and held-out scores use merged LoRA.
LoRA requires activation memory: a small adapter does not make training as cheap
as cached head fitting. The runner never silently reduces batch size on OOM.
448 defaults for an RTX 4060 Laptop with 8 GiB VRAM / 16 GiB RAM:
LoRA microbatch 1, accumulation 16, eval batch 1, benchmark batch 2.
These settings require on-device memory validation; no fit guarantee.
BATCH_SIZE describes inference; MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS describes training.
Final partial windows and ignored segmentation pixels use exact valid-target normalization.


Run the notebook's existing BaselineModel definition first, then call:
    run_baselines(BaselineModel, ROOT=Path.cwd(),
                  EPOCHS={"classification": 3, "segmentation": 5})

In cached_head mode, train/dev features are built fresh. Train stores both RGB
orientations; dev stores the original. Cache arrays are float32 disk-backed NPY.
No cross-run cache reuse or automatic training resume is performed. Completed frozen-head
checkpoints are saved each epoch; restarting this function starts a new run.
Cached training is ONLY for a deterministic, entirely frozen backbone, and the
original mean-pool classification / spatial segmentation head interfaces.
Full-model held-out evaluation, FLOPs and timings always use real RGB images.
Cache extraction + head fit costs are recorded separately and together.
Importing this file never starts training. For scripts guard calls with
if __name__ == "__main__": . Notebook factories remain in the parent process.

Metric guide (all accuracy values are percentages):
  heldout_top1_pct / top5_pct: classification quality; Top-1 is primary.
  heldout_miou_pct: mean class intersection/union; segmentation primary.
  heldout_pixel_accuracy_pct: overall pixel correctness; frequent classes dominate.
  classes_with_nonzero_union: denominator of mIoU; absent classes are excluded.
  train/dev/heldout_loss: cross-entropy learning/overfitting diagnostic. ADE loss
    uses valid pixels only; do not compare its magnitude to classification loss.
  whole_model_forward_gflops_per_image: primary algorithmic compute estimate.
  attention_forward_gflops_per_image: attention-only compute diagnostic.
    Both use 2 FLOPs/MAC, include supported dense matmul/conv/SDPA, and exclude
    elementwise work, softmax, normalization and data movement. Unknown coverage
    yields null, never a fabricated total. GMACs = GFLOPs / 2 in this convention.
  FLOP batch mean/std/min/max: input-dependent compute variation, NOT seed variance.
  forward[_backward]_partial_gflops_per_image: profiler diagnostic only; NOT a
    substitute for audited whole-model counts. Backward excludes optimizer work.
  b1/bB latency median/p95 (ms per BATCH): typical/tail device-stream delay.
  b1/bB images_per_second: CUDA-stream throughput. wall_images_per_second also
    includes host launch overhead. Both exclude data loading and input transfer.
  throughput std: repeated timing trials, NOT accuracy confidence intervals.
  peak_allocated_gib: peak live CUDA tensors (includes resident model/inputs).
  peak_reserved_gib: allocator pool size, including reusable cached blocks.
  attention/mlp/head/other GPU time and percentages: bottleneck diagnostics;
    attribution unavailable => percentages null. These are NOT FLOP shares.
  train_seconds / train_images_per_second / peak memory: adaptation cost.
  cache time/bytes/backbone forwards: frozen-mode extraction and storage cost.
  fit/heldout/experiment seconds: pipeline budgeting; scopes differ explicitly.
  backbone/head/LoRA parameter counts: model size and adaptation capacity.
  epochs, LRs, images, optimizer steps, best dev epoch/score, gradient norms,
    split/source/base hashes and environment: audit/reproducibility metadata.

Suggested main results: Top-1 or mIoU + whole-model GFLOPs + batch-1 latency +
batch-B wall throughput + peak allocated memory. Keep remaining diagnostics in
appendices. Same hardware, precision, resolution, batch and training budget are
needed for comparisons. A single training seed does not establish significance.
ADE masks here are resized to SIZE, not standard full-resolution ADE evaluation.
Reference: ToMe (arxiv.org/abs/2210.09461) reports throughput alongside accuracy;
LoRA (arxiv.org/abs/2106.09685) merges updates into weights for inference.
"""
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import types
import os
import platform
import random
import shutil
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm.auto import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    # Top-level, importable function for Windows spawn.
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _tf32_settings(enabled):
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    if hasattr(matmul, "fp32_precision"):
        targets = (matmul, cudnn.conv, cudnn.rnn)
        key, value = "fp32_precision", "tf32" if enabled else "ieee"
    else:
        targets = (matmul, cudnn)
        key, value = "allow_tf32", bool(enabled)
    return targets, key, value


def _set_tf32(enabled):
    targets, key, value = _tf32_settings(enabled)
    for obj in targets:
        setattr(obj, key, value)


@contextmanager
def evaluation_tf32(enabled):
    """Scope TF32 to evaluation; preserve the caller's precision API and flags."""
    targets, key, value = _tf32_settings(enabled)
    previous = [getattr(obj, key) for obj in targets]
    try:
        for obj in targets:
            setattr(obj, key, value)
        yield
    finally:
        for obj, saved in zip(targets, previous):
            setattr(obj, key, saved)


class _LoRATrainingForward:
    """Compile a callable, never wrap the model or change checkpoint keys.

    Include backward in the retry boundary: AOT compilation is lazy. Restore
    accumulated gradients and RNG before eager retry; optimizer steps stay outside.
    """
    def __init__(self, model, enabled=False, backend="inductor", mode="default", report=None):
        self.model, self.forward, self.report = model, model, report
        self.info = dict(requested=enabled, backend=backend, mode=mode,
                         active=False, error=None, torch=str(torch.__version__),
                         python=platform.python_version(), platform=platform.platform(),
                         triton=package_version("triton") or package_version("triton-windows"))
        if enabled:
            try:
                self.forward = torch.compile(model.forward, backend=backend, mode=mode)
                self.info["active"] = True
            except Exception as exc:
                self.disable(exc)
        self.record()

    def record(self):
        if self.report is not None:
            write_json(self.report, self.info)

    def disable(self, exc):
        self.forward = self.model
        self.info.update(active=False, error=f"{type(exc).__name__}: {exc}")
        print("LoRA compilation failed; retaining eager execution:", self.info["error"])
        self.record()

    def backward(self, images, target, task, normalizer, autocast):
        def step():
            with autocast():
                logits = self.forward(images, task)
            loss, count = loss_sum_and_count(logits.float(), target, task)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("Non-finite training logits/loss; optimizer not updated.")
            if count:
                (loss / normalizer).backward()
            return logits, loss, count

        if not self.info["active"]:
            return step()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        try:
            return step()
        except Exception as exc:
            for p, grad in zip(parameters, gradients):
                p.grad = grad
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            self.disable(exc)
        return step()


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(obj, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


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


def load_baseline_checkpoint(model, checkpoint, *, merge=True):
    """Restore head and, for LoRA, adapters against the verified original weights.

    Construct model with the original pretrained weights first.  No automatic
    training resume is provided.  merge=True is intended for inference.
    """
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("training_mode") == "lora":
        if _frozen_backbone_digest(model.backbone) != checkpoint["pretrained_backbone_sha256"]:
            raise ValueError("Pretrained backbone differs from the LoRA checkpoint base.")
        spec = checkpoint["lora"]
        inject_lora(model, rank=spec["rank"], alpha=spec["alpha"], targets=spec["targets"],
                    unwrap_backbone_forward=spec["unwrapped_backbone_forward"])
        _load_adapter_state(model.backbone, checkpoint["lora_state_dict"])
        if merge:
            merge_lora(model)
    getattr(model, f"{checkpoint['task']}_head").load_state_dict(checkpoint["head_state_dict"], strict=True)
    return model


def _check_gradients(model, parameters, max_norm):
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("A frozen parameter received a gradient.")
    # error_if_nonfinite rejects NaN/Inf BEFORE the optimizer changes weights.
    return float(nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True))


def _finish_accumulated_step(model, optimizer, parameters, normalizer, valid_count, max_norm):
    """Normalize accumulated summed gradients BEFORE checking/clipping/updating."""
    if valid_count <= 0:
        raise ValueError("Accumulated optimizer step needs valid targets.")
    scale = normalizer / valid_count
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
    norm = _check_gradients(model, parameters, max_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


def write_csv(path, rows):
    if rows:
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with open(temporary, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class SavedSplitDataset(Dataset):
    """All worker dependencies are instance data, not notebook globals."""
    def __init__(self, task, part, entries, root, size, class_to_idx, val_labels):
        self.task, self.part = task, part
        self.entries, self.root, self.size = entries, Path(root), size
        self.class_to_idx, self.val_labels = class_to_idx, val_labels

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.get_item(index, self.part == "train" and random.random() < 0.5)

    def get_item(self, index, flip):
        image, target = self.decode_item(index, flip)
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div_(255)
        if self.task != "classification":
            target = torch.from_numpy(target).long()
        return image, target

    def decode_item(self, index, flip):
        """Pillow/NumPy only: safe to call without entering Torch in a worker."""
        # Explicit flip avoids sharing random state between loading threads.
        entry = self.entries[index]
        image_path = entry if self.task == "classification" else entry["image"]
        with Image.open(self.root / image_path) as im:
            image = im.convert("RGB").resize(
                (self.size, self.size), Image.Resampling.BICUBIC
            )
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = np.array(image, dtype=np.uint8, copy=True)
        if self.task == "classification":
            label = (self.val_labels[Path(entry).stem] if self.part == "val"
                     else self.class_to_idx[Path(entry).parent.name])
            self._report_loading(index)
            return image, label
        with Image.open(self.root / entry["mask"]) as im:
            mask = im.resize((self.size, self.size), Image.Resampling.NEAREST)
        if flip:
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        target = np.array(mask, dtype=np.int64, copy=True)
        if target.ndim != 2 or target.min() < 0 or target.max() > 150:
            raise ValueError(f"Unexpected ADE20K mask: {entry['mask']}")
        target = np.where(target == 0, 255, target - 1)
        self._report_loading(index)
        return image, target

    def _report_loading(self, index):
        # Installed only during a sequential dev probe in this process.
        progress = getattr(self, "_loading_progress", None)
        if progress is not None and index < progress.total:
            progress.update(1)


@contextmanager
def first_batch_progress(loader):
    """Real image counts for threads/zero workers; batch count for processes."""
    per_image = isinstance(loader, ThreadedImageLoader) or loader.num_workers == 0
    total = min(loader.batch_size, len(loader.dataset)) if per_image else 1
    # Plain tqdm renders reliably in notebook text output as well as terminals.
    from tqdm import tqdm as text_progress
    with text_progress(total=total, desc="First dev batch: images decoded" if per_image
                       else "First dev batch: waiting for workers", unit="image" if per_image
                       else "batch", mininterval=0.1, leave=True) as progress:
        if per_image:
            loader.dataset._loading_progress = progress
        try:
            yield progress, per_image
        finally:
            if per_image:
                del loader.dataset._loading_progress


@contextmanager
def loader_timer(label, interval=5.0):
    """Report wall time, including while blocked; diagnostic, not a timeout."""
    started = time.perf_counter()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(interval):
            tqdm.write(f"{label}: still waiting ({time.perf_counter() - started:.1f}s)")

    print(f"{label}...", flush=True)
    reporter = threading.Thread(target=heartbeat, name="loader-timer", daemon=True)
    reporter.start()
    status = "finished"
    try:
        yield
    except BaseException:
        status = "interrupted/failed"
        raise
    finally:
        stopped.set()
        reporter.join()
        print(f"{label}: {status} in {time.perf_counter() - started:.2f}s", flush=True)


def timed_image_batches(loader, label):
    """Measure consumer wait for CPU batches, not GPU time or total worker work.

    Prefetch overlaps training. Update one bar without adding CUDA synchronization.
    """
    iterator = iter(loader)
    batches = 0
    total = maximum = 0.0
    progress = tqdm(total=len(loader), desc=label, unit="batch", mininterval=1.0)
    try:
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            elapsed = time.perf_counter() - started
            batches += 1
            total += elapsed
            maximum = max(maximum, elapsed)
            progress.set_postfix_str(f"load mean={total / batches:.3f}s", refresh=False)
            yield batch
            progress.update(1)
    finally:
        if hasattr(iterator, "close"):
            iterator.close()
        progress.close()


class ThreadedImageLoader:
    """Bounded, ordered CPU image prefetch without Windows child interpreters.

    NUM_WORKERS counts threads for this backend. Each pending task is one batch;
    at most num_workers batches are prefetched. Random choices happen in the
    iterator thread, so completion order cannot change shuffle or augmentation.
    """
    def __init__(self, dataset, batch_size, num_workers, shuffle, seed, pin_memory=True):
        self.dataset, self.batch_size = dataset, batch_size
        self.num_workers, self.shuffle = num_workers, shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.pin_memory = pin_memory

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _batch(self, choices):
        samples = []
        for index, flip in choices:
            try:
                samples.append(self.dataset.decode_item(index, flip))
            except Exception as exc:
                raise RuntimeError(
                    f"Image loading failed: {self.dataset.task}/{self.dataset.part} "
                    f"entry={self.dataset.entries[index]!r}: {exc}") from exc
        return samples

    def __iter__(self):
        indices = (torch.randperm(len(self.dataset), generator=self.generator).tolist()
                   if self.shuffle else list(range(len(self.dataset))))
        flips = (torch.rand(len(indices), generator=self.generator).lt(0.5).tolist()
                 if self.dataset.part == "train" else [False] * len(indices))
        batches = iter([list(zip(indices[i:i+self.batch_size], flips[i:i+self.batch_size]))
                        for i in range(0, len(indices), self.batch_size)])
        pool = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="image-loader")
        pending = deque()
        try:
            for _ in range(self.num_workers):
                choices = next(batches, None)
                if choices is not None:
                    pending.append(pool.submit(self._batch, choices))
            while pending:
                samples = pending.popleft().result()
                choices = next(batches, None)
                if choices is not None:
                    pending.append(pool.submit(self._batch, choices))
                # All Torch conversion, collation and pinning stay in the consumer.
                images, targets = zip(*samples)
                images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().float().div_(255)
                targets = (torch.tensor(targets, dtype=torch.long)
                           if self.dataset.task == "classification" else
                           torch.from_numpy(np.stack(targets)).long())
                batch = [images, targets]
                if self.pin_memory:
                    batch = [tensor.pin_memory() for tensor in batch]
                yield batch
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)


def loss_sum_and_count(logits, target, task):
    if task == "classification":
        return F.cross_entropy(logits, target, reduction="sum"), target.numel()
    count = int((target != 255).sum().item())
    if count == 0:
        return logits.sum() * 0, 0
    return F.cross_entropy(logits, target, ignore_index=255, reduction="sum"), count


@contextmanager
def component_labels(model, task):
    originals, selected = [], []
    matched = {"attention": [], "mlp": [], "head": []}
    for name, module in model.backbone.named_modules():
        leaf = name.rsplit(".", 1)[-1].lower()
        kind = ("attention" if leaf in {"attn", "self_attn", "attention"}
                else "mlp" if leaf in {"mlp", "ffn"} else None)
        if kind and not any(name.startswith(parent + ".") for parent in selected):
            selected.append(name)
            matched[kind].append(name)
            originals.append((module, module.forward, kind))
    head = getattr(model, f"{task}_head")
    originals.append((head, head.forward, "head"))
    matched["head"].append(f"{task}_head")

    def wrapped(original, label):
        def forward(*args, **kwargs):
            with record_function(f"component/{label}"):
                return original(*args, **kwargs)
        return forward

    try:
        for module, original, kind in originals:
            module.forward = wrapped(original, kind)
        yield matched
    finally:
        for module, original, _ in originals:
            module.forward = original


# Attention counts use the dense-matmul convention: one multiply-add = 2 FLOPs.
# These lists make deliberate exclusions visible instead of claiming every
# arithmetic instruction is counted.  An unrecognized operation blocks the
# comparable summary value until its coverage is reviewed.
_ATTENTION_LAYOUT_OPS = {
    # Embedding gathers table rows; no matrix/conv arithmetic is performed.
    'embedding',
    'alias', 'as_strided', 'clone', 'contiguous', 'copy_', 'detach', 'empty',
    'empty_like', 'empty_strided', 'expand', 'flatten', 'index', 'index_select',
    'lift_fresh', 'new_empty', 'new_empty_strided', 'permute', 'reshape', 'select',
    'slice', 'split', 'split_with_sizes', 'squeeze', 'stack', 't', 'to',
    'transpose', 'unbind', 'unflatten', 'unsqueeze', 'view', '_unsafe_view',
    '_to_copy', 'cat', 'gather', 'scatter', 'scatter_', 'repeat',
    'repeat_interleave', 'narrow', 'zeros', 'zeros_like', 'ones', 'ones_like',
    'full', 'full_like', 'fill_', 'zero_', 'arange', 'scalar_tensor',
    '_local_scalar_dense', 'is_nonzero', 'resolve_conj', 'resolve_neg',
}
_ATTENTION_EXCLUDED_ARITHMETIC = {
    'add', 'add_', 'sub', 'sub_', 'mul', 'mul_', 'div', 'div_', 'neg', 'pow',
    'exp', 'log', 'sqrt', 'rsqrt', 'sin', 'cos', 'softmax', '_softmax',
    '_safe_softmax', 'log_softmax', '_log_softmax', 'native_layer_norm',
    'layer_norm', 'rms_norm', 'native_group_norm', 'group_norm', 'dropout',
    'native_dropout', 'bernoulli', 'bernoulli_', 'rand', 'rand_like', 'randn',
    'uniform_', 'normal_', 'masked_fill', 'masked_fill_', 'where', 'tril',
    'triu', 'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'logical_not', 'logical_and',
    'logical_or', 'bitwise_not', 'bitwise_and', 'bitwise_or', 'sum', 'mean',
    'amax', 'amin', 'max', 'min', 'var', 'var_mean', 'clamp', 'clamp_min',
    'clamp_max', 'relu', 'gelu', 'sigmoid', 'silu', 'tanh', 'abs',
    'linalg_vector_norm', 'norm', 'topk', 'sort', 'argsort', 'argmax',
    'argmin', 'scatter_add', 'scatter_add_', 'scatter_reduce',
}


def _find_attention_modules(model, module_names=None):
    """Return non-overlapping attention roots, relative to model.backbone."""
    modules = dict(model.backbone.named_modules())
    if module_names is not None:
        names = list(module_names)
        if len(set(names)) != len(names) or any(n not in modules for n in names):
            raise ValueError('Attention module names must be unique existing backbone paths.')
        names.sort(key=lambda n: (n.count('.'), n))
        for i, name in enumerate(names):
            if any(p == '' or name.startswith(p + '.') for p in names[:i]):
                raise ValueError('Attention module paths must not overlap (would double count).')
    else:
        names = []
        for name, module in modules.items():
            leaf = name.rsplit('.', 1)[-1].lower()
            if (leaf in {'attn', 'self_attn', 'attention'} or
                    'attention' in type(module).__name__.lower()):
                if not any(p == '' or name.startswith(p + '.') for p in names):
                    names.append(name)
    if not names:
        raise ValueError('No attention modules found.  Supply module_names relative to model.backbone.')
    return {n: modules[n] for n in names}


def _sdpa_dense_flops(q_shape, k_shape, v_shape, *args, **kwargs):
    """Dense QK^T + AV only, BHND layout.  Masked attention is a dense estimate."""
    if not all(len(s) == 4 for s in (q_shape, k_shape, v_shape)):
        raise ValueError('Expected BHND SDPA tensors; refusing an ambiguous attention count.')
    b, h, nq, dq = q_shape
    bk, hk, nk, dk = k_shape
    bv, hv, nv, dv = v_shape
    if not (b == bk == bv and dq == dk and nk == nv and hk == hv and h % hk == 0):
        raise ValueError('Unexpected SDPA tensor dimensions.')
    return int(2 * b * h * nq * nk * (dq + dv))


def measure_forward_flops(model, images, task, directory=None,
                            module_names=None, custom_mapping=None):
    """Whole-model and attention counts from the same eval forward.

    Returns whole-model plus attention summary fields. Optional directory gets
    audited operator ledgers and a per-module CSV. Counts include projections inside
    selected attention roots and supported attention products, using 2 FLOPs/MAC.
    Softmax, bias additions, normalization, activations, masking, sorting and
    memory movement are excluded.  Thus this is NOT an all-instruction count.
    Masking a dense attention matrix does not reduce this count.  Sparse/custom
    kernels need explicit formulas, supplied through custom_mapping using the
    FlopCounterMode API; audit those formulas against the implemented algorithm.
    module_names is a list of paths relative to model.backbone, useful when new
    attention implementations use different names.  Code must run eagerly.
    """
    from torch.utils.flop_counter import FlopCounterMode
    from torch.utils._python_dispatch import TorchDispatchMode

    if hasattr(model, '_orig_mod') or hasattr(model.backbone, '_orig_mod'):
        raise ValueError('Count eager model operations; torch.compile can hide custom kernels.')
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError('images must be a nonempty [B, C, H, W] tensor.')
    selected = _find_attention_modules(model, module_names)
    # Explicit formulas for standard BHND fused SDPA variants, independent of
    # profiler CUDA kernel attribution and of version-specific registry gaps.
    mapping = {}
    for name in ('_scaled_dot_product_flash_attention',
                 '_scaled_dot_product_flash_attention_for_cpu',
                 '_scaled_dot_product_efficient_attention',
                 '_scaled_dot_product_cudnn_attention'):
        if hasattr(torch.ops.aten, name):
            mapping[getattr(torch.ops.aten, name)] = _sdpa_dense_flops
    mapping.update(custom_mapping or {})
    counter = FlopCounterMode(display=False, custom_mapping=mapping)
    # Freeze the counted operator families across PyTorch versions.
    allowed = {'mm', 'addmm', 'bmm', 'baddbmm', '_scaled_mm',
               'convolution', '_convolution', 'cudnn_convolution',
               '_slow_conv2d_forward', 'convolution_overrideable'}
    counter.flop_registry = {
        op: formula for op, formula in counter.flop_registry.items()
        if op in mapping or (str(op).startswith('aten.') and
                             str(op).rsplit('.', 1)[-1] in allowed)
    }
    ledger = {name: {'calls': 0, 'flops': 0, 'by_operator': Counter(),
                     'observed_operators': Counter(), 'operator_input_shapes': {},
                     'input_shapes': []}
              for name in selected}
    active = []
    global_observed = Counter()

    class AuditMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            global_observed[str(func._overloadpacket)] += 1
            if active:
                entry = ledger[active[-1]]
                op = str(func._overloadpacket)
                entry['observed_operators'][op] += 1
                if func._overloadpacket in counter.flop_registry:
                    shapes = [list(x.shape) for x in args if isinstance(x, torch.Tensor)]
                    shapes += [list(x.shape) for x in (kwargs or {}).values()
                               if isinstance(x, torch.Tensor)]
                    signatures = entry['operator_input_shapes'].setdefault(op, [])
                    if shapes not in signatures:
                        signatures.append(shapes)
            return func(*args, **(kwargs or {}))

    def snapshot():
        return dict(counter.get_flop_counts().get('Global', {}))

    def wrap(original, name):
        def forward(*args, **kwargs):
            before = snapshot()
            entry = ledger[name]
            entry['calls'] += 1
            shapes = [list(x.shape) for x in args if isinstance(x, torch.Tensor)]
            shapes += [list(x.shape) for x in kwargs.values() if isinstance(x, torch.Tensor)]
            if shapes not in entry['input_shapes']:
                entry['input_shapes'].append(shapes)
            active.append(name)
            try:
                return original(*args, **kwargs)
            finally:
                active.pop()
                for op, value in snapshot().items():
                    delta = int(value - before.get(op, 0))
                    if delta:
                        entry['by_operator'][str(op)] += delta
                        entry['flops'] += delta
        return forward

    # Preserve exact per-submodule eval/train flags and original forward lookup.
    training_flags = [(m, m.training) for m in model.modules()]
    originals = []
    try:
        model.eval()
        for name, module in selected.items():
            originals.append((module, 'forward' in module.__dict__,
                              module.__dict__.get('forward')))
            module.forward = wrap(module.forward, name)
        with torch.inference_mode(False), torch.no_grad(), counter, AuditMode():
            output = model(images, task)
            del output
    finally:
        for module, had_instance_forward, old in reversed(originals):
            if had_instance_forward:
                module.forward = old
            else:
                delattr(module, 'forward')
        for module, flag in training_flags:
            module.training = flag

    registered = {str(k) for k in counter.flop_registry}
    unknown = Counter()
    excluded = Counter()
    layout = Counter()
    rows = []
    for name, entry in ledger.items():
        for op, calls in entry['observed_operators'].items():
            leaf = op.rsplit('.', 1)[-1]
            if op in registered or op in entry['by_operator']:
                continue
            if op.startswith('aten.') and leaf in _ATTENTION_LAYOUT_OPS:
                layout[op] += calls
            elif op.startswith('aten.') and leaf in _ATTENTION_EXCLUDED_ARITHMETIC:
                excluded[op] += calls
            else:
                unknown[op] += calls
        rows.append({
            'module': name, 'calls': entry['calls'],
            'counted_forward_flops_per_image': entry['flops'] / images.shape[0],
            'counted_forward_gflops_per_image': entry['flops'] / images.shape[0] / 1e9,
            'parameters': sum(p.numel() for p in selected[name].parameters()),
            'trainable_parameters': sum(p.numel() for p in selected[name].parameters() if p.requires_grad),
        })
    missed = [n for n, e in ledger.items() if not e['calls'] or not e['flops']]
    coverage_ok = not unknown and not missed
    unique_parameters = {id(p): p for m in selected.values() for p in m.parameters()}
    counted = sum(e['flops'] for e in ledger.values()) / images.shape[0] / 1e9
    global_counts = {str(op): int(v) for op, v in snapshot().items()}
    global_unknown, global_excluded, global_layout = Counter(), Counter(), Counter()
    # Conventional matrix/conv FLOPs deliberately omit interpolation/pooling,
    # scalar arithmetic, normalization and data movement.  Record all omissions.
    model_exclusions = _ATTENTION_EXCLUDED_ARITHMETIC | {
        'upsample_bilinear2d', 'upsample_bicubic2d', 'upsample_nearest2d',
        '_upsample_bilinear2d_aa', '_upsample_bicubic2d_aa',
        'adaptive_avg_pool2d', '_adaptive_avg_pool2d', 'avg_pool2d',
        'max_pool2d_with_indices', 'constant_pad_nd', 'reflection_pad2d',
        'replication_pad2d', 'pad', 'batch_norm', 'native_batch_norm',
        '_native_batch_norm_legit_no_training', 'leaky_relu', 'hardswish',
    }
    for op, calls in global_observed.items():
        leaf = op.rsplit('.', 1)[-1]
        if op in registered or op in global_counts:
            continue
        if op.startswith('aten.') and leaf in _ATTENTION_LAYOUT_OPS:
            global_layout[op] += calls
        elif op.startswith('aten.') and leaf in model_exclusions:
            global_excluded[op] += calls
        else:
            global_unknown[op] += calls
    whole = sum(global_counts.values()) / images.shape[0] / 1e9
    whole_ok = not global_unknown and whole > 0
    if counted > whole + 1e-8:
        raise RuntimeError('Attention count exceeds whole-model count; possible overlapping scopes.')
    summary = {
        'whole_model_forward_gflops_per_image': whole if whole_ok else None,
        'whole_model_forward_gmacs_per_image': whole / 2 if whole_ok else None,
        'whole_model_counted_partial_gflops_per_image': whole,
        'whole_model_flops_coverage_ok': whole_ok,
        'whole_model_flops_unmapped_ops': ', '.join(sorted(global_unknown)),
        'whole_model_flops_convention': 'dense_matmul_conv_2_flops_per_mac',
        'whole_model_flops_counter_version': 3,
        'attention_forward_gflops_per_image': counted if coverage_ok else None,
        'attention_counted_partial_gflops_per_image': counted,
        'attention_flops_coverage_ok': coverage_ok,
        'attention_flops_convention': 'dense_matmul_2_flops_per_mac',
        'attention_flops_counter_version': 3,
        'attention_flops_unmapped_ops': ', '.join(sorted(unknown)),
        'attention_flops_uncounted_modules': ', '.join(missed),
        'attention_module_count': len(selected),
        'attention_parameters': sum(p.numel() for p in unique_parameters.values()),
        'attention_trainable_parameters': sum(p.numel() for p in unique_parameters.values() if p.requires_grad),
    }
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'whole_model_flops.json', {
            **summary, 'input_shape': list(images.shape), 'task': task,
            'torch_version': torch.__version__, 'passes': 1,
            'scope': 'Complete model(images, task), including active task head.',
            'by_operator_flops_per_batch': global_counts,
            'observed_operators': global_observed,
            'unmapped_operators': global_unknown,
            'excluded_arithmetic_operators': global_excluded,
            'layout_and_memory_operators': global_layout,
            'notes': [
                '2 FLOPs per MAC; counts matrix products, convolutions and dense SDPA products.',
                'Includes supported scorer/merge/reconstruction products anywhere in the model.',
                'Excludes softmax, normalization, activation, interpolation, pooling, sorting and elementwise arithmetic.',
                'Coverage means coverage of this convention, not every arithmetic instruction.',
                'Masks do not imply sparse execution; custom kernels require audited custom_mapping formulas.',
                'GMACs = GFLOPs / 2 under this convention; check each paper before comparing.',
                'Only executed paths are audited. Opaque external CUDA calls require manual review.',
            ],
        })
        write_json(directory / 'attention_flops.json', {
            **summary, 'torch_version': torch.__version__,
            'input_shape': list(images.shape), 'task': task,
            'passes': 1, 'mode': 'eval_forward_only',
            'scope': 'Matched attention modules in backbone; includes their internal projections.',
            'matched_modules': list(selected), 'by_module': ledger,
            'unmapped_operators': unknown, 'excluded_arithmetic_operators': excluded,
            'layout_and_memory_operators': layout,
            'custom_formula_operators': [str(k) for k in mapping],
            'notes': [
                'One multiply-add counts as 2 FLOPs.  Values are per image.',
                'Includes registered dense matrix/conv products and SDPA QK^T + AV.',
                'Excludes softmax, bias, normalization, activations, masking, sorting and data movement.',
                'coverage_ok only refers to this declared convention in the matched scopes.',
                'Unknown kernels or unexecuted modules make the comparable summary null; inspect the ledger.',
                'Masked SDPA is counted as dense work; zeros or masks alone do not imply sparse execution.',
                'Compression/routing outside matched attention modules is not included.',
                'For dynamic token counts, sample multiple real images and report distribution statistics.',
                'This does not measure attention backward or optimizer work.',
                'Counts are algorithmic estimates, not hardware instruction or time measurements.',
            ],
        })
        write_csv(directory / 'attention_flops_by_module.csv', rows)
    return summary


def measure_attention_flops(*args, **kwargs):
    """Compatibility alias; now also returns whole-model forward counts."""
    return measure_forward_flops(*args, **kwargs)


def measure_flops_on_dataset(model, dataset, task, batch_size, batches, seed,
                             directory, module_names=None, custom_mapping=None):
    """Deterministic dev subset; reports mean and variation of batch averages."""
    n = min(len(dataset), batch_size * batches)
    indices = np.random.default_rng(seed).choice(len(dataset), size=n, replace=False)
    device = next(model.parameters()).device
    directory = Path(directory)
    samples = []
    weights = []
    for offset in range(0, n, batch_size):
        chosen = indices[offset:offset + batch_size]
        images = torch.stack([dataset[int(i)][0] for i in chosen]).to(device)
        sample = measure_forward_flops(
            model, images, task, directory / 'flop_samples' / f'batch_{len(samples):03d}',
            module_names=module_names, custom_mapping=custom_mapping)
        samples.append(sample)
        weights.append(len(chosen))
        del images
    summary = dict(samples[0])
    for prefix in ('whole_model', 'attention'):
        ok = all(row[f'{prefix}_flops_coverage_ok'] for row in samples)
        summary[f'{prefix}_flops_coverage_ok'] = ok
        key = f'{prefix}_forward_gflops_per_image'
        partial = f'{prefix}_counted_partial_gflops_per_image'
        vals = np.array([row[partial] for row in samples])
        mean = float(np.average(vals, weights=weights))
        summary[partial] = mean
        summary[key] = mean if ok else None
        summary[f'{prefix}_forward_gflops_batch_mean_std'] = (
            float(vals.std(ddof=1)) if ok and len(vals) > 1 else (0.0 if ok else None))
        summary[f'{prefix}_forward_gflops_batch_mean_min'] = float(vals.min()) if ok else None
        summary[f'{prefix}_forward_gflops_batch_mean_max'] = float(vals.max()) if ok else None
        summary[f'{prefix}_flops_unmapped_ops'] = ', '.join(sorted({
            op for row in samples for op in row[f'{prefix}_flops_unmapped_ops'].split(', ') if op}))
    summary['attention_flops_uncounted_modules'] = ', '.join(sorted({
        op for row in samples for op in row['attention_flops_uncounted_modules'].split(', ') if op}))
    total = summary['whole_model_forward_gflops_per_image']
    summary['whole_model_forward_gmacs_per_image'] = total / 2 if total is not None else None
    summary.update(flops_sample_images=n, flops_sample_batches=len(samples), flops_sample_seed=seed)
    write_json(directory / 'forward_flops.json', {
        **summary, 'dataset_indices': indices.tolist(), 'batch_sizes': weights,
        'samples': samples,
        'notes': ['Std/min/max describe batch-average GFLOPs/image, not per-image variation.',
                  'Raw operators, exclusions and unknowns are recorded in flop_samples/batch_*/.',
                  'No legacy profiler partial estimate is used as a replacement for missing coverage.'],
    })
    return summary




def _component_time_summary(prof, matched):
    """Partition exclusive CPU-attributed CUDA time by enclosing component.

    Count CPU-side self device time only, never GPU events as well: adding both
    would double-count the same kernels. Missing attribution remains unavailable.
    """
    durations = dict.fromkeys(("attention", "mlp", "head", "other"), 0.0)
    negative_events = 0
    for event in prof.events():
        if str(getattr(event, "device_type", "")).split(".")[-1] != "CPU":
            continue
        scope = event
        component = None
        inside = False
        while scope is not None:
            key = getattr(scope, "name", "")
            if component is None and key in {"component/attention", "component/mlp", "component/head"}:
                component = key.split("/")[-1]
            if key == "component/total":
                inside = True
                break
            scope = getattr(scope, "cpu_parent", None)
        if not inside:
            continue
        value = getattr(event, "self_device_time_total", None)
        if value is None:
            value = getattr(event, "self_cuda_time_total", 0.0)
        value = float(value)
        if value < -1e-6:
            negative_events += 1
        durations[component or "other"] += max(value, 0.0)
    total = sum(durations.values())
    reasons = []
    if total <= 0:
        reasons.append("No CUDA time attributed to CPU operations inside the forward scope.")
    if negative_events:
        reasons.append("Negative exclusive CUDA times; profiler attribution is inconsistent.")
    for kind in ("attention", "mlp", "head"):
        if not matched[kind]:
            reasons.append(f"No matched {kind} modules.")
        elif durations[kind] <= 0:
            reasons.append(f"No CUDA time attributed to matched {kind} modules.")
    ok = not reasons
    return {
        "component_gpu_attribution_available": ok,
        "component_timing_method": "profiler_exclusive_cuda_time_by_cpu_scope_v2",
        "component_gpu_attribution_reason": "; ".join(reasons),
        "component_total_gpu_us": total,
        **{f"{k}_gpu_time_us": v for k, v in durations.items()},
        **{f"{k}_gpu_time_pct": 100 * v / total if ok else None
           for k, v in durations.items()},
    }


class CachedFeatureDataset(Dataset):
    """Memory-map features lazily; each returned array is an owned writable copy.

    Cache loaders deliberately use zero workers: small pooled features do not
    benefit from Windows spawn and dense features otherwise multiply RAM use.
    """
    def __init__(self, directory, training=False):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "cache.json").read_text())
        if not self.meta.get("complete"):
            raise ValueError(f"Incomplete feature cache: {directory}")
        self.training = training
        if training and self.meta["views"] != 2:
            raise ValueError("Training cache must contain both orientations.")
        self._features = self._targets = None

    def __len__(self):
        return self.meta["images"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = state["_targets"] = None
        return state

    def __getitem__(self, index):
        if self._features is None:
            self._features = np.load(self.directory / "features.npy", mmap_mode="r")
            self._targets = np.load(self.directory / "targets.npy", mmap_mode="r")
        flip = self.training and random.random() < 0.5
        view = int(flip)
        features = torch.from_numpy(np.array(self._features[index, view], copy=True))
        if self.meta["task"] == "classification":
            return features, int(self._targets[index])
        target = self._targets[index]
        if flip:
            target = np.flip(target, axis=-1)
        return features, torch.from_numpy(np.array(target, dtype=np.int64, copy=True))

    def close(self):
        for key in ("_features", "_targets"):
            array = getattr(self, key)
            if array is not None:
                array._mmap.close()
                setattr(self, key, None)


class _HeadFromCache(nn.Module):
    def __init__(self, head, task, grid, size):
        super().__init__()
        self.head, self.task = head, task
        self.grid, self.size = tuple(grid), size

    def forward(self, features, task):
        if task != self.task:
            raise ValueError("Cached head task mismatch.")
        if task == "classification":
            # Stored [B,1,D]; the original head's mean over one token is identity.
            return self.head(features)
        return self.head(features, self.grid, (self.size, self.size))


def _validate_cache_tensor(actual, expected, *, label, report, atol=1e-4):
    """Check float32 equivalence; record diagnostics even when validation fails."""
    row = {"label": label, "rtol": 1e-4, "atol": atol,
           "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    report.append(row)
    if actual.shape != expected.shape:
        row["passed"] = False
        raise RuntimeError(f"{label}: cache shape mismatch.")
    if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        row["passed"] = False
        raise RuntimeError(f"{label}: nonfinite cache validation values.")
    error = (actual - expected).abs()
    row["max_abs_error"] = float(error.max().item())
    row["mean_abs_error"] = float(error.mean().item())
    row["passed"] = bool(torch.all(error <= atol + 1e-4 * expected.abs()).item())
    if not row["passed"]:
        raise RuntimeError(f"{label}: cache validation failed: {row}")


def _cache_features(patches, task):
    if not isinstance(patches, torch.Tensor) or patches.ndim != 3:
        raise ValueError("Expected backbone patches [B,N,D].")
    if not torch.isfinite(patches).all():
        raise RuntimeError("Nonfinite backbone features.")
    return patches.float().mean(1, keepdim=True) if task == "classification" else patches.float()


@torch.no_grad()
def build_feature_cache(model, dataset, task, part, directory, *,
                        device, batch_size, num_workers, seed, loader_backend="process"):
    """One original + one flipped backbone forward per training image.

    dataset must disable random augmentation. Targets are stored once; the
    corresponding segmentation target is flipped when a flipped view is read.
    """
    if dataset.part == "train":
        raise ValueError("Extraction dataset must disable random augmentation.")
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("Cannot cache a trainable backbone.")
    model.eval()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    views = 2 if part == "train" else 1
    worker_args = {"multiprocessing_context": "spawn", "prefetch_factor": 1} if num_workers else {}
    if num_workers and loader_backend == "thread":
        loader = ThreadedImageLoader(dataset, batch_size, num_workers, False, seed,
                                     pin_memory=device.type == "cuda")
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=device.type == "cuda",
                            worker_init_fn=seed_worker, persistent_workers=False,
                            generator=torch.Generator().manual_seed(seed), **worker_args)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    feature_array = target_array = None
    offset = 0
    meta = {"complete": False, "task": task, "part": part,
            "images": len(dataset), "views": views, "dtype": "float32",
            "size": dataset.size, "augmentation": "both horizontal orientations"}
    write_json(directory / "cache.json", meta)
    try:
        for images, targets in timed_image_batches(loader, f"Cache {task}/{part}"):
            images = images.to(device, non_blocking=True)
            patches, grid = model.backbone(images)
            features = _cache_features(patches, task)
            grid = tuple(int(x) for x in grid)
            shape = tuple(features.shape[1:])
            if task == "segmentation" and patches.shape[1] != grid[0] * grid[1]:
                raise ValueError("Spatial tokens must match the returned grid.")
            if feature_array is None:
                feature_shape = (len(dataset), views, *shape)
                target_shape = (len(dataset), *targets.shape[1:])
                target_dtype = np.int64 if task == "classification" else np.uint8
                required = (int(np.prod(feature_shape)) * 4 +
                            int(np.prod(target_shape)) * np.dtype(target_dtype).itemsize)
                free = shutil.disk_usage(directory).free
                if free < required + 2 * 2**30:
                    raise RuntimeError(
                        f"Cache needs {required / 2**30:.1f} GiB plus 2 GiB free; "
                        f"only {free / 2**30:.1f} GiB available. Choose another CACHE_ROOT."
                    )
                print(f"  {part} cache: {required / 2**30:.2f} GiB; float32")
                feature_array = np.lib.format.open_memmap(
                    directory / "features.npy", mode="w+", dtype=np.float32,
                    shape=feature_shape)
                target_array = np.lib.format.open_memmap(
                    directory / "targets.npy", mode="w+", dtype=target_dtype,
                    shape=target_shape)
                meta.update(grid=list(grid), feature_shape=list(shape), bytes=required)
            if list(grid) != meta["grid"] or list(shape) != meta["feature_shape"]:
                raise ValueError("Cache requires a fixed feature grid and feature dimension.")
            end = offset + images.shape[0]
            feature_array[offset:end, 0] = features.cpu().numpy()
            target_array[offset:end] = targets.numpy()
            if views == 2:
                flipped_patches, flipped_grid = model.backbone(images.flip(-1))
                flipped = _cache_features(flipped_patches, task)
                if tuple(flipped.shape) != tuple(features.shape) or tuple(flipped_grid) != grid:
                    raise ValueError("Flipped features changed shape.")
                feature_array[offset:end, 1] = flipped.cpu().numpy()
                del flipped, flipped_patches
            offset = end
            del images, patches, features, targets
        if offset != len(dataset) or feature_array is None:
            raise RuntimeError("Cache image count mismatch.")
        feature_array.flush()
        target_array.flush()
        if is_cuda:
            torch.cuda.synchronize(device)
        meta.update(complete=True, seconds=time.perf_counter() - start,
                    backbone_image_forwards=offset * views,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if is_cuda else 0.0,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if is_cuda else 0.0)
        write_json(directory / "cache.json", meta)
    finally:
        for array in (feature_array, target_array):
            if array is not None:
                array._mmap.close()
    return meta


class _Runner:
    def __init__(self, model_factory, config):
        self.model_factory = model_factory
        self.c = dict(config)
        self.c.setdefault("TRAIN_PRECISION", "fp32")
        self.c.setdefault("LOADER_BACKEND", "process")
        self.c.setdefault("FAIL_FAST", True)
        self.device = torch.device("cuda:0")
        self.root = config["ROOT"]
        self.roots = {"classification": config["IMAGENET_ROOT"],
                      "segmentation": config["ADE_ROOT"]}
        self.split_paths = {
            "classification": config["SPLIT_ROOT"] / "imagenet_split.json",
            "segmentation": config["SPLIT_ROOT"] / "ade20k_split.json",
        }

    def prepare(self):
        c = self.c
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable.  Select your CUDA PyTorch notebook kernel.")
        if c["STRICT_DETERMINISM"]:
            if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
                raise RuntimeError("Strict determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization.")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(c["STRICT_DETERMINISM"])
        torch.cuda.set_device(self.device)
        if c.get("TRAIN_PRECISION", "fp32") == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported by this GPU/PyTorch build. Set TRAIN_PRECISION='fp32'.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        _set_tf32(False)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.splits = {task: json.loads(path.read_text(encoding="utf-8"))
                       for task, path in self.split_paths.items()}
        self.class_to_idx = self.splits["classification"]["class_to_idx"]
        if (len(self.class_to_idx) != 1000 or
                set(self.class_to_idx.values()) != set(range(1000))):
            raise ValueError("ImageNet class_to_idx must map 1000 classes to 0..999.")
        self.val_labels = {}
        with open(c["IMAGENET_ROOT"] / "LOC_val_solution.csv", newline="",
                  encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                tokens = row["PredictionString"].split()
                image_id = row["ImageId"]
                if not tokens or len(tokens) % 5:
                    raise ValueError(f"Malformed localization label: {image_id}")
                synsets = set(tokens[::5])
                if len(synsets) != 1 or not synsets.issubset(self.class_to_idx):
                    raise ValueError(f"Ambiguous/unknown classification label: {image_id}: {synsets}")
                if image_id in self.val_labels:
                    raise ValueError(f"Duplicate validation label: {image_id}")
                self.val_labels[image_id] = self.class_to_idx[next(iter(synsets))]
        for task, split in self.splits.items():
            seen = set()
            for part in ("train", "dev", "val"):
                entries = split[f"{part}_files"]
                if not entries:
                    raise ValueError(f"Empty {task}/{part} split")
                for entry in tqdm(entries, desc=f"Validate {task}/{part}", unit="files"):
                    rel = entry if task == "classification" else entry["image"]
                    canonical = os.path.normcase(str((self.roots[task] / rel).resolve()))
                    if canonical in seen:
                        raise ValueError(f"Duplicate or overlapping split entry: {rel}")
                    seen.add(canonical)
                    paths = [self.roots[task] / rel]
                    if task == "segmentation":
                        paths.append(self.roots[task] / entry["mask"])
                    for path in paths:
                        if not path.is_file():
                            raise FileNotFoundError(path)
                    if task == "classification":
                        if part == "val" and Path(rel).stem not in self.val_labels:
                            raise ValueError(f"Missing validation label: {rel}")
                        if part != "val" and Path(rel).parent.name not in self.class_to_idx:
                            raise ValueError(f"Unknown training class: {rel}")
            print(task, {p: len(split[f"{p}_files"]) for p in ("train", "dev", "val")})
        self.out = self.root / "report 1"/ "baseline measurments" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.out.mkdir(parents=True, exist_ok=False)
        write_json(self.out / "configuration.json", {
            "training_mode": c["TRAINING_MODE"],
            "cache_dtype": "float32" if c["TRAINING_MODE"] == "cached_head" else None,
            "lora": {"rank": c["LORA_R"], "alpha": c["LORA_ALPHA"], "lr": c["LORA_LR"],
                     "targets": c["LORA_TARGETS"], "dropout": 0.0,
                     "unwrap_backbone_forward": c["LORA_UNWRAP_BACKBONE_FORWARD"]},
            "max_grad_norm": c["MAX_GRAD_NORM"] if math.isfinite(c["MAX_GRAD_NORM"]) else None,
            "strict_determinism": c["STRICT_DETERMINISM"],
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cache_root": str(c["CACHE_ROOT"]), "cache_batch_size": c["CACHE_BATCH_SIZE"],
            "keep_cache": c["KEEP_CACHE"], "cached_loader_workers": 0,
            "training_seconds_scope": ("train feature extraction + cached head epochs" if c["TRAINING_MODE"] == "cached_head" else "uncached RGB adapter + head training"),
            "seed": c["SEED"], "input_size": c["SIZE"], "batch_size": c["BATCH_SIZE"],
            "micro_batch_size": c["MICRO_BATCH_SIZE"], "grad_accum_steps": c["GRAD_ACCUM_STEPS"],
            "effective_train_batch_size": c["MICRO_BATCH_SIZE"] * c["GRAD_ACCUM_STEPS"],
            "eval_batch_size": c["EVAL_BATCH_SIZE"],
            "batch_size_scope": "inference benchmark; LoRA training uses micro_batch_size",
            "accumulation_loss_normalization": "total valid targets per window, including final partial window",
            "epochs": c["EPOCHS"], "learning_rates": c["LR"],
            "weight_decay": c["WEIGHT_DECAY"], "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR", "precision": "inference float32; TF32 disabled",
            "train_precision": c["TRAIN_PRECISION"], "evaluation_precision": "fp32",
            "evaluation_tf32": c.get("EVAL_TF32", False),
            "compile_lora": c.get("COMPILE_LORA", False),
            "compile_backend": c.get("COMPILE_BACKEND", "inductor"),
            "compile_mode": c.get("COMPILE_MODE", "default"),
            "benchmark_precision": "fp32", "training_profile_precision": c["TRAIN_PRECISION"],
            "loader_backend": c["LOADER_BACKEND"], "fail_fast": c["FAIL_FAST"],
            "frozen_backbones": c["TRAINING_MODE"] == "cached_head",
            "pretrained_parameters_frozen": True,
            "augmentation": f"resize to {c['SIZE']}x{c['SIZE']}; training horizontal flip p=0.5",
            "segmentation_evaluation": f"{c['SIZE']}x{c['SIZE']} masks, 150 classes, ignore label 255",
            "classification_head": "mean patch pooling + linear",
            "segmentation_head": "1x1 conv + GN + GELU + 3x3 conv + GN + GELU + 1x1 conv",
            "heldout_split": "official validation; used after dev-based selection",
            "imagenet_heldout_labels": "LOC_val_solution.csv, unique synset validated",
            "split_sha256": {t: hashlib.sha256(p.read_bytes()).hexdigest()
                             for t, p in self.split_paths.items()},
            "split_sizes": {t: {p: len(s[f'{p}_files']) for p in ('train', 'dev', 'val')}
                            for t, s in self.splits.items()},
            "gpu": torch.cuda.get_device_name(self.device),
            "gpu_total_gib": torch.cuda.get_device_properties(self.device).total_memory / 2**30,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            **{p: package_version(p) for p in ("torchvision", "transformers", "timm")},
            "python": platform.python_version(), "num_workers": c["NUM_WORKERS"],
            "multiprocessing_context": "spawn" if c["NUM_WORKERS"] and c["LOADER_BACKEND"] == "process" else None,
            "persistent_workers": False,
            "flop_batches": c["FLOP_BATCHES"], "bench_trials": c["BENCH_TRIALS"],
            "whole_model_flops_convention": "dense_matmul_conv_2_flops_per_mac",
            "flops_counter_version": 3,
            "bench_warmup": c["BENCH_WARMUP"], "bench_repeats": c["BENCH_REPEATS"],
            "profile_repeats": c["PROFILE_REPEATS"], "model_names": c["MODEL_NAMES"],
            "tasks": c["TASKS"], "cache_validation_version": 2,
            "attention_module_names": c["ATTENTION_MODULE_NAMES"],
            "attention_flops_convention": "dense_matmul_2_flops_per_mac",
        })
        print("\nGPU:", torch.cuda.get_device_name(self.device))
        print("Results:", self.out)
        print("Training precision:", c["TRAIN_PRECISION"], "| evaluation TF32:", c.get("EVAL_TF32", False),
              "| benchmarks: fp32, TF32 disabled")
        print("Image loader:", c["LOADER_BACKEND"], "| workers:", c["NUM_WORKERS"])

    def make_loader(self, task, part):
        c = self.c
        dataset = SavedSplitDataset(
            task, part, self.splits[task][f"{part}_files"], self.roots[task],
            c["SIZE"], self.class_to_idx if task == "classification" else {},
            self.val_labels if task == "classification" and part == "val" else {},
        )
        batch_size = (c["MICRO_BATCH_SIZE"] if part == "train" else c["EVAL_BATCH_SIZE"]) if c["TRAINING_MODE"] == "lora" else c["BATCH_SIZE"]
        if c["NUM_WORKERS"] and c.get("LOADER_BACKEND", "process") == "thread":
            return ThreadedImageLoader(dataset, batch_size, c["NUM_WORKERS"],
                                       part == "train", c["SEED"])
        worker_args = {"multiprocessing_context": "spawn", "prefetch_factor": 1} if c["NUM_WORKERS"] else {}
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=part == "train",
            num_workers=c["NUM_WORKERS"], pin_memory=True, drop_last=False,
            generator=torch.Generator().manual_seed(c["SEED"]),
            worker_init_fn=seed_worker, persistent_workers=False, **worker_args,
        )

    def training_autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.c.get("TRAIN_PRECISION", "fp32") == "bf16")

    @torch.inference_mode()
    def evaluate(self, model, loader, task):
        model.eval()
        loss_total = count_total = correct1 = correct5 = 0
        confusion = torch.zeros((150, 150), dtype=torch.int64, device=self.device)
        for images, target in timed_image_batches(loader, f"Evaluate {task}"):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with evaluation_tf32(self.c.get("EVAL_TF32", False)):
                logits = model(images, task)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite evaluation logits.")
            loss_sum, count = loss_sum_and_count(logits, target, task)
            loss_total += loss_sum.item()
            count_total += count
            if task == "classification":
                predictions = logits.topk(5, dim=1).indices
                correct1 += (predictions[:, 0] == target).sum().item()
                correct5 += (predictions == target[:, None]).any(dim=1).sum().item()
            else:
                valid = target != 255
                prediction = logits.argmax(dim=1)
                encoded = target[valid] * 150 + prediction[valid]
                confusion += torch.bincount(encoded, minlength=150 * 150).reshape(150, 150)
        if count_total == 0:
            raise ValueError("Evaluation split has no valid labels.")
        result = {"loss": loss_total / count_total}
        if task == "classification":
            result.update(top1_pct=100 * correct1 / count_total, top5_pct=100 * correct5 / count_total)
        else:
            cm = confusion.double()
            intersection = cm.diag()
            union = cm.sum(0) + cm.sum(1) - intersection
            present = union > 0
            result.update(
                miou_pct=100 * (intersection[present] / union[present]).mean().item(),
                pixel_accuracy_pct=100 * intersection.sum().item() / cm.sum().item(),
                classes_with_nonzero_union=int(present.sum().item()),
            )
        return result

    @torch.inference_mode()
    def benchmark(self, model, images, task, directory=None):
        model.eval()
        result, trial_rows = {}, []
        for batch in dict.fromkeys((1, self.c["BATCH_SIZE"])):
            tag = f"b{batch}"
            x = images[:batch].contiguous()
            if len(x) != batch:
                raise ValueError('Benchmark input is smaller than requested batch.')
            gpu_trials, wall_trials, peaks, all_times = [], [], [], []
            for trial in range(self.c["BENCH_TRIALS"]):
                for _ in range(self.c["BENCH_WARMUP"]):
                    model(x, task)
                torch.cuda.synchronize()
                # Allocate and initialize events outside timed work.
                pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                         for _ in range(self.c["BENCH_REPEATS"])]
                for start, end in pairs:
                    start.record()
                    end.record()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                for start, end in pairs:
                    start.record()
                    model(x, task)
                    end.record()
                torch.cuda.synchronize()
                times = np.array([a.elapsed_time(b) for a, b in pairs])
                peak = torch.cuda.max_memory_allocated() / 2**30
                # Separate uninstrumented wall-clock loop includes host overhead.
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(self.c["BENCH_REPEATS"]):
                    model(x, task)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                gpu_rate = float(batch * 1000 / times.mean())
                wall_rate = batch * self.c["BENCH_REPEATS"] / seconds
                gpu_trials.append(gpu_rate)
                wall_trials.append(wall_rate)
                peaks.append(peak)
                all_times.extend(times.tolist())
                trial_rows.append(dict(batch_size=batch, trial=trial + 1,
                    gpu_images_per_second=gpu_rate, wall_images_per_second=wall_rate,
                    gpu_latency_median_ms=float(np.median(times)),
                    gpu_latency_p95_ms=float(np.percentile(times, 95)),
                    wall_mean_batch_ms=seconds * 1000 / self.c["BENCH_REPEATS"],
                    gpu_latency_samples_ms=times.tolist(), peak_allocated_gib=peak))
            result.update({
                f"{tag}_latency_median_ms": float(np.median(all_times)),
                f"{tag}_latency_p95_ms": float(np.percentile(all_times, 95)),
                f"{tag}_images_per_second": float(np.mean(gpu_trials)),
                f"{tag}_images_per_second_std": float(np.std(gpu_trials, ddof=1)) if len(gpu_trials)>1 else None,
                f"{tag}_wall_images_per_second": float(np.mean(wall_trials)),
                f"{tag}_wall_images_per_second_std": float(np.std(wall_trials, ddof=1)) if len(wall_trials)>1 else None,
                f"{tag}_peak_allocated_gib": max(peaks),
            })
        result['benchmark_trials'] = self.c['BENCH_TRIALS']
        if directory is not None:
            write_json(Path(directory) / 'benchmark_trials.json', {
                **result, 'trials': trial_rows,
                'notes': ['Same checkpoint and resident GPU inputs across timing trials; no retraining.',
                          'Excludes loading, preprocessing and host-to-device transfer.',
                          'Wall throughput includes Python/CPU launch overhead and final synchronization.',
                          'CUDA-event latency describes GPU-stream elapsed time, not end-to-end request latency.',
                          'Repeated trials use one representative batch. Dynamic input-dependent methods need broader input sampling.'],
            })
        return result

    def profile_model(self, model, images, target, task, directory, profile_backward=True):
        repeats = self.c["PROFILE_REPEATS"]
        model.eval()
        with component_labels(model, task) as matched:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True, with_flops=True) as prof:
                with torch.no_grad():
                    for _ in range(repeats):
                        with record_function("component/total"):
                            model(images, task)
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(directory / "inference_trace.json"))
        averages = prof.key_averages()
        result = _component_time_summary(prof, matched)
        result["forward_partial_gflops_per_image"] = sum(float(e.flops or 0) for e in averages) / repeats / images.shape[0] / 1e9
        if profile_backward:
            model.train()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True, with_flops=True) as train_prof:
                for _ in range(repeats):
                    model.zero_grad(set_to_none=True)
                    with record_function("training/forward_and_head_backward"):
                        with self.training_autocast():
                            logits = model(images, task)
                        loss_sum, count = loss_sum_and_count(logits.float(), target, task)
                        (loss_sum / max(count, 1)).backward()
                torch.cuda.synchronize()
            train_prof.export_chrome_trace(str(directory / "training_trace.json"))
            result["forward_backward_partial_gflops_per_image"] = sum(float(e.flops or 0) for e in train_prof.key_averages()) / repeats / images.shape[0] / 1e9
            model.zero_grad(set_to_none=True)
            model.eval()
        write_json(directory / "profiling.json", {
            **result, "matched_modules": matched, "batch_size": images.shape[0],
            "notes": [
                "Percentages describe profiled forward GPU time, not FLOP shares.",
                "Attention includes QKV/output projections inside attention modules.",
                "Profiler overhead is excluded from benchmark speed measurements.",
                "FLOPs are partial supported-operation estimates, not total compute.",
                "Training trace scope is recorded separately for LoRA; frozen mode profiles head backward only.",
                "Inference benchmark excludes loading, preprocessing and host-to-device transfer.",
            ],
        })
        return result

    def run_lora_experiment(self, name, task):
        c = self.c
        experiment_start = time.perf_counter()
        seed_everything(c['SEED'])
        gc.collect()
        torch.cuda.empty_cache()
        directory = self.out / f'{task}_{name}'
        directory.mkdir()
        loaders = {p: self.make_loader(task, p) for p in ('train', 'dev', 'val')}
        with loader_timer("Checking image loader before model allocation"), first_batch_progress(loaders['dev']) as (progress, per_image):
            probe_iterator = iter(loaders['dev'])
            try:
                with loader_timer("Loading first dev batch"):
                    check_images, check_target = next(probe_iterator)
                    if not per_image:
                        progress.update(1)
            finally:
                with loader_timer("Closing image loader check"):
                    if hasattr(probe_iterator, "close"):
                        probe_iterator.close()
                    del probe_iterator
        model = self.model_factory(name)
        inactive = 'segmentation' if task == 'classification' else 'classification'
        setattr(model, f'{inactive}_head', nn.Identity())
        model = model.to(self.device).eval()
        head = getattr(model, f'{task}_head')
        head.requires_grad_(True)
        base_hash = _frozen_backbone_digest(model.backbone)
        if len(loaders['dev'].dataset) < c['BATCH_SIZE']:
            raise ValueError('Dev split must contain one full benchmark batch.')
        probe = check_images[:1].to(self.device)
        with torch.no_grad():
            initial = model(probe, task)
        expected = ((len(probe), 1000) if task == 'classification' else
                    (len(probe), 150, c['SIZE'], c['SIZE']))
        if tuple(initial.shape) != expected:
            raise ValueError(f'Unexpected model output: {initial.shape}; expected {expected}')
        spec = inject_lora(model, rank=c['LORA_R'], alpha=c['LORA_ALPHA'],
                           targets=c['LORA_TARGETS'].get(name),
                           unwrap_backbone_forward=c['LORA_UNWRAP_BACKBONE_FORWARD'])
        adapters = [p for p in model.backbone.parameters() if p.requires_grad]
        parameters = list(head.parameters()) + adapters
        allowed = {id(p) for p in parameters}
        if {id(p) for p in model.parameters() if p.requires_grad} != allowed:
            raise RuntimeError('Unexpected trainable parameters outside adapters and active head.')
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(probe, task), initial, rtol=0, atol=0)
        # A synthetic scalar checks connectivity even for all-ignore ADE masks.
        # B starts at zero: A's gradient may be zero, but must not be absent.
        model.zero_grad(set_to_none=True)
        with self.training_autocast():
            output = model(probe, task)
        if not output.requires_grad:
            raise RuntimeError('Model forward is under no_grad/inference_mode; LoRA cannot train.')
        output.float().square().mean().backward()
        disconnected = [f'{n}.{key}' for n, m in _lora_modules(model.backbone).items()
                        for key in ('lora_A', 'lora_B') if getattr(m, key).grad is None]
        if disconnected:
            raise RuntimeError(f'LoRA gradients disconnected: {disconnected}.  A nested no_grad, detach, or unused projection blocks training.')
        _check_gradients(model, parameters, c['MAX_GRAD_NORM'])
        if not any(torch.count_nonzero(m.lora_B.grad).item() for m in _lora_modules(model.backbone).values()):
            raise RuntimeError('All initial LoRA B gradients are zero; refusing silent head-only training.')
        model.zero_grad(set_to_none=True)
        del output, initial, probe
        if _frozen_backbone_digest(model.backbone) != base_hash:
            raise RuntimeError('Pretrained state changed during adapter setup.')
        write_json(directory / 'lora_setup.json', {**spec, 'pretrained_backbone_sha256': base_hash,
                   'initial_output_equivalence': True, 'gradient_connectivity': True,
                   'cache_enabled': False, 'backbone_eval_mode': True})
        initial_adapters = _adapter_state(model.backbone)
        optimizer = torch.optim.AdamW([
            {'params': list(head.parameters()), 'lr': c['LR'][task], 'name': 'head'},
            {'params': adapters, 'lr': c['LORA_LR'], 'name': 'lora'},
        ], weight_decay=c['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c['EPOCHS'][task])
        metric = 'top1_pct' if task == 'classification' else 'miou_pct'
        best_score, best_epoch = -float('inf'), None
        history = []
        total_steps = total_images = 0
        fit_start = time.perf_counter()
        training_forward = _LoRATrainingForward(
            model, c.get('COMPILE_LORA', False), c.get('COMPILE_BACKEND', 'inductor'),
            c.get('COMPILE_MODE', 'default'), directory / 'compile_status.json')
        for epoch in range(1, c['EPOCHS'][task] + 1):
            seed_everything(c['SEED'] + epoch)
            loaders['train'].generator.manual_seed(c['SEED'] + epoch)
            model.train()
            model.backbone.eval()  # eval controls stochastic layers, not autograd.
            head.train()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            loss_total = count_total = epoch_images = epoch_steps = 0
            max_grad_norm = 0.0
            window_count = window_batches = 0
            # A fixed denominator keeps backward magnitudes modest.  At the
            # update boundary rescale by the ACTUAL number of valid targets.
            normalizer = c['MICRO_BATCH_SIZE'] * c['GRAD_ACCUM_STEPS']
            if task == 'segmentation':
                normalizer *= c['SIZE'] ** 2
            optimizer.zero_grad(set_to_none=True)
            for images, target in timed_image_batches(loaders['train'], f'{task}/{name} LoRA epoch {epoch}'):
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                logits, loss_sum, count = training_forward.backward(
                    images, target, task, normalizer, self.training_autocast)
                window_count += count
                window_batches += 1
                loss_total += loss_sum.detach().item()
                count_total += count
                epoch_images += len(images)
                # Release the previous graph/logits before the next forward.
                del logits, loss_sum, images, target
                if window_batches == c['GRAD_ACCUM_STEPS']:
                    if window_count:
                        norm = _finish_accumulated_step(model, optimizer, parameters,
                                                       normalizer, window_count, c['MAX_GRAD_NORM'])
                        max_grad_norm = max(max_grad_norm, norm)
                        epoch_steps += 1
                    else:
                        optimizer.zero_grad(set_to_none=True)
                    window_count = window_batches = 0
            # Flush the last short window; all-ignore windows have no update.
            if window_count:
                norm = _finish_accumulated_step(model, optimizer, parameters,
                                               normalizer, window_count, c['MAX_GRAD_NORM'])
                max_grad_norm = max(max_grad_norm, norm)
                epoch_steps += 1
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - started
            allocated = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            if not count_total or not epoch_steps:
                raise RuntimeError('No valid training targets/optimizer steps.')
            optimizer.zero_grad(set_to_none=True)
            if _frozen_backbone_digest(model.backbone) != base_hash:
                raise RuntimeError('Frozen pretrained parameters or buffers changed during training.')
            current_adapters = _adapter_state(model.backbone)
            if not any(not torch.equal(v, initial_adapters[k]) for k, v in current_adapters.items()):
                raise RuntimeError('Adapters did not change; refusing silent head-only training.')
            total_steps += epoch_steps
            total_images += epoch_images
            started = time.perf_counter()
            dev = self.evaluate(model, loaders['dev'], task)
            dev_seconds = time.perf_counter() - started
            row = {'training_mode': 'lora', 'train_precision': c['TRAIN_PRECISION'], 'evaluation_precision': 'fp32', 'epoch': epoch,
                   'compile_active': training_forward.info['active'], 'evaluation_tf32': c.get('EVAL_TF32', False),
                   'lr': optimizer.param_groups[0]['lr'], 'lora_lr': optimizer.param_groups[1]['lr'],
                   'train_loss': loss_total / count_total, 'train_seconds': train_seconds,
                   'train_images_per_second': epoch_images / train_seconds,
                   'train_images': epoch_images, 'optimizer_steps': epoch_steps,
                   'micro_batch_size': c['MICRO_BATCH_SIZE'], 'grad_accum_steps': c['GRAD_ACCUM_STEPS'],
                   'effective_train_batch_size': c['MICRO_BATCH_SIZE'] * c['GRAD_ACCUM_STEPS'],
                   'train_peak_allocated_gib': allocated, 'train_peak_reserved_gib': reserved,
                   'max_preclip_grad_norm': max_grad_norm, 'dev_seconds': dev_seconds,
                   **{f'dev_{k}': v for k, v in dev.items()}}
            history.append(row)
            write_csv(directory / 'epochs.csv', history)
            checkpoint = {'model_name': name, 'task': task, 'epoch': epoch,
                          'compile_options': dict(training_forward.info),
                          'evaluation_tf32': c.get('EVAL_TF32', False),
                          'feature_dim': model.backbone.feature_dim, 'input_size': c['SIZE'],
                          'head_state_dict': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                          'lora_state_dict': current_adapters, 'lora': spec,
                          'pretrained_backbone_sha256': base_hash, 'training_mode': 'lora', 'train_precision': c['TRAIN_PRECISION'], 'evaluation_precision': 'fp32',
                          'dev_metrics': dev, 'seed': c['SEED'], 'checkpoint_format_version': 1,
                          'resume_supported': False}
            atomic_torch_save(checkpoint, directory / 'last_lora.pt')
            if dev[metric] > best_score:
                best_score, best_epoch = dev[metric], epoch
                atomic_torch_save(checkpoint, directory / 'best_lora.pt')
            scheduler.step()
            print(f"  train loss={row['train_loss']:.4f} | dev {metric}={dev[metric]:.2f} | train={train_seconds / 60:.1f} min | peak={allocated:.2f} GiB")
        fit_seconds = time.perf_counter() - fit_start
        training_forward.record()
        compile_options = dict(training_forward.info)
        del training_forward
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scheduler, parameters, adapters
        if 'loss' in locals():
            del loss
        checkpoint = torch.load(directory / 'best_lora.pt', map_location='cpu', weights_only=True)
        head.load_state_dict(checkpoint['head_state_dict'])
        _load_adapter_state(model.backbone, checkpoint['lora_state_dict'])
        # Training profiling uses the training microbatch, never the inference batch.
        profile_images, profile_target = next(iter(loaders['train']))
        images = profile_images.to(self.device)
        target = profile_target.to(self.device)
        del profile_images, profile_target
        # Profile adaptation before merging, so backward includes the adapters.
        compute = {}
        try:
            compute.update(self.profile_lora_training(model, images, target, task, directory))
        except Exception as exc:
            compute['training_profiling_error'] = str(exc)
            (directory / 'training_profiling_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        finally:
            model.zero_grad(set_to_none=True)
            model.eval()
        del images, target
        # Keep the merge audit at one image; segmentation logits can be large.
        images = check_images[:1].to(self.device)
        with torch.no_grad():
            before = model(images, task)
        merge_lora(model)
        with torch.no_grad():
            after = model(images, task)
            if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                raise RuntimeError("Non-finite outputs during LoRA merge validation.")

            torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-3)
            merge_error = float((after - before).abs().max())
        write_json(directory / 'lora_merge_validation.json', {
            'passed': True, 'max_absolute_logit_error': merge_error,
            'rtol': 1e-4, 'atol': 1e-3, 'images_checked': len(images),
            'note': 'Dev-batch numerical check; held-out metrics use merged model.'})
        del before, after, checkpoint, current_adapters, initial_adapters
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        started = time.perf_counter()
        heldout = self.evaluate(model, loaders['val'], task)
        test_seconds = time.perf_counter() - started
        del images
        benchmark_samples = [loaders['dev'].dataset[i] for i in range(c['BATCH_SIZE'])]
        images = torch.stack([sample[0] for sample in benchmark_samples]).to(self.device)
        target = torch.stack([torch.as_tensor(sample[1]) for sample in benchmark_samples]).to(self.device)
        del benchmark_samples
        speed = self.benchmark(model, images, task, directory)
        try:
            compute.update(measure_flops_on_dataset(
                model, loaders['dev'].dataset, task, c['BATCH_SIZE'], c['FLOP_BATCHES'],
                c['SEED'], directory, module_names=c['ATTENTION_MODULE_NAMES'].get(name)))
        except Exception as exc:
            compute.update(whole_model_flops_coverage_ok=False, whole_model_forward_gflops_per_image=None,
                           attention_flops_coverage_ok=False, attention_forward_gflops_per_image=None,
                           flops_error=str(exc))
            (directory / 'attention_flops_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        try:
            compute.update(self.profile_model(model, images, target, task, directory, profile_backward=False))
        except Exception as exc:
            compute['profiling_error'] = str(exc)
            (directory / 'profiling_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        profiling_path = directory / 'profiling.json'
        profiling = json.loads(profiling_path.read_text(encoding='utf-8')) if profiling_path.exists() else {}
        write_json(profiling_path, {**profiling, **compute, 'inference_lora_merged': True,
                                   'profile_training_scope': 'unmerged RGB forward + adapter/head backward; no optimizer'})
        result = {'model': name, 'task': task, 'status': 'completed', 'seed': c['SEED'],
                  'compile_options': compile_options, 'evaluation_tf32': c.get('EVAL_TF32', False),
                  'resolution': c['SIZE'], 'batch_size': c['BATCH_SIZE'], 'epochs': c['EPOCHS'][task],
                  'micro_batch_size': c['MICRO_BATCH_SIZE'], 'grad_accum_steps': c['GRAD_ACCUM_STEPS'],
                  'effective_train_batch_size': c['MICRO_BATCH_SIZE'] * c['GRAD_ACCUM_STEPS'],
                  'eval_batch_size': c['EVAL_BATCH_SIZE'],
                  'best_epoch': best_epoch, 'best_dev_metric': best_score, 'selection_metric': metric,
                  'training_mode': 'lora', 'train_precision': c['TRAIN_PRECISION'], 'evaluation_precision': 'fp32', 'cache_enabled': False, 'inference_lora_merged': True,
                  'lora_rank': spec['rank'], 'lora_alpha': spec['alpha'], 'lora_lr': c['LORA_LR'],
                  'trainable_lora_parameters': spec['trainable_parameters'],
                  'backbone_parameters': sum(p.numel() for p in model.backbone.parameters()),
                  'trainable_head_parameters': sum(p.numel() for p in head.parameters()),
                  'total_training_images': total_images, 'total_optimizer_steps': total_steps,
                  'training_seconds': sum(r['train_seconds'] for r in history),
                  'training_seconds_scope': 'RGB loading + forward + adapter/head backward + optimizer; excludes dev and safety audits',
                  'fit_seconds_including_dev_and_saves': fit_seconds,
                  'heldout_evaluation_seconds': test_seconds,
                  'train_peak_allocated_gib': max(r['train_peak_allocated_gib'] for r in history),
                  'train_peak_reserved_gib': max(r['train_peak_reserved_gib'] for r in history),
                  'inference_scope': 'merged full RGB model; no feature cache',
                  'profile_training_scope': 'unmerged RGB forward + adapter/head backward; no optimizer',
                  'pretrained_backbone_sha256': base_hash, 'frozen_state_verified': True,
                  'merge_max_absolute_logit_error': merge_error,
                  'best_checkpoint_path': str(directory / 'best_lora.pt'),
                  'experiment_seconds_including_measurements': time.perf_counter() - experiment_start,
                  **{f'heldout_{k}': v for k, v in heldout.items()}, **speed, **compute}
        write_json(directory / 'result.json', result)
        return result

    def profile_lora_training(self, model, images, target, task, directory):
        repeats = self.c['PROFILE_REPEATS']
        model.train()
        model.backbone.eval()
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True, with_flops=True) as prof:
                for _ in range(repeats):
                    model.zero_grad(set_to_none=True)
                    with record_function('training/forward_and_lora_head_backward'):
                        with self.training_autocast():
                            logits = model(images, task)
                        loss, count = loss_sum_and_count(logits.float(), target, task)
                        if not count:
                            raise ValueError('No valid labels in training profile batch.')
                        (loss / count).backward()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(directory / 'training_trace.json'))
            return {'forward_backward_partial_gflops_per_image':
                    sum(float(e.flops or 0) for e in prof.key_averages()) / repeats / len(images) / 1e9,
                    'training_profile_lora_merged': False,
                    'training_profile_precision': self.c.get('TRAIN_PRECISION', 'fp32'),
                    'training_profile_batch_size': len(images)}
        finally:
            model.zero_grad(set_to_none=True)
            model.eval()

    def run_experiment(self, name, task):
        if self.c["TRAINING_MODE"] == "lora":
            return self.run_lora_experiment(name, task)
        c = self.c
        experiment_start = time.perf_counter()
        seed_everything(c["SEED"])
        gc.collect()
        torch.cuda.empty_cache()
        directory = self.out / f"{task}_{name}"
        directory.mkdir()
        loaders = {p: self.make_loader(task, p) for p in ("train", "dev", "val")}
        model = self.model_factory(name)
        inactive = "segmentation" if task == "classification" else "classification"
        setattr(model, f"{inactive}_head", nn.Identity())
        model = model.to(self.device)
        head = getattr(model, f"{task}_head")
        if any(p.requires_grad for p in model.backbone.parameters()):
            raise RuntimeError("Cached-head mode requires an entirely frozen backbone.")
        optimizer = torch.optim.AdamW(head.parameters(), lr=c["LR"][task], weight_decay=c["WEIGHT_DECAY"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c["EPOCHS"][task])
        check_images, check_target = next(iter(loaders["dev"]))
        model.eval()
        with torch.no_grad():
            check_logits = model(check_images[:1].to(self.device), task)
        expected = (1, 1000) if task == "classification" else (1, 150, c["SIZE"], c["SIZE"])
        if tuple(check_logits.shape) != expected:
            raise ValueError(f"{name}: output {tuple(check_logits.shape)} != expected {expected}")
        del check_logits
        # Validate the cached-head interface and both input orientations before
        # spending time creating large caches. Trainable heads may not pool
        # tokens differently from the original notebook's ClassificationHead.
        cache_validation = []
        for flip in (False, True):
            x = check_images[:2].to(self.device)
            if flip:
                x = x.flip(-1)
            with torch.no_grad():
                patches, grid = model.backbone(x)
                features = _cache_features(patches, task)
                cached_model = _HeadFromCache(head, task, grid, c["SIZE"])
                cached_model.eval()
                try:
                    _validate_cache_tensor(
                        cached_model(features, task), model(x, task),
                        label=f"head_interface_flip_{flip}", report=cache_validation)
                finally:
                    write_json(directory / "cache_validation.json", cache_validation)
            del x, patches, features
        raw_dev_dataset = loaders["dev"].dataset
        cache_dir = c["CACHE_ROOT"] / self.out.name / f"{task}_{name}"
        cache_dir.mkdir(parents=True, exist_ok=False)
        cache_stats = {}
        for part in ("train", "dev"):
            # Using 'dev' disables random flips; train/dev labels both come
            # from ImageNet class folders. Extraction is sequential.
            raw_dataset = SavedSplitDataset(
                task, "dev", self.splits[task][f"{part}_files"],
                self.roots[task], c["SIZE"], self.class_to_idx, {})
            cache_stats[part] = build_feature_cache(
                model, raw_dataset, task, part, cache_dir / part,
                device=self.device, batch_size=c["CACHE_BATCH_SIZE"],
                num_workers=c["NUM_WORKERS"], seed=c["SEED"],
                loader_backend=c["LOADER_BACKEND"])
            if tuple(cache_stats[part]["grid"]) != tuple(grid):
                raise ValueError("Extraction grid differs from shape-check grid.")
            dataset = CachedFeatureDataset(cache_dir / part, training=part == "train")
            loaders[part] = DataLoader(
                dataset, batch_size=c["BATCH_SIZE"], shuffle=part == "train",
                num_workers=0, pin_memory=True, drop_last=False,
                generator=torch.Generator().manual_seed(c["SEED"]))
        write_json(directory / "feature_cache.json", {
            "path": str(cache_dir), "parts": cache_stats,
            "cache_policy": "fresh per experiment; no cross-run reuse",
            "keep_cache": c["KEEP_CACHE"],
            "note": "Caching preserves both horizontal views, not bitwise GPU reproducibility."})
        # Replay the FIRST extraction batch at its original batch size. GPU
        # kernels can round differently when extraction uses B images but the
        # replay uses only two. Check stored FEATURES before checking the head.
        replay_loader = DataLoader(
            raw_dev_dataset, batch_size=c["CACHE_BATCH_SIZE"], shuffle=False,
            num_workers=0, generator=torch.Generator().manual_seed(c["SEED"]))
        replay_images, replay_targets = next(iter(replay_loader))
        ncheck = min(2, len(replay_images))
        stored = [loaders["dev"].dataset[i] for i in range(ncheck)]
        cached_x = torch.stack([item[0] for item in stored]).to(self.device)
        cached_y = torch.stack([torch.as_tensor(item[1]) for item in stored])
        try:
            with torch.no_grad():
                torch.testing.assert_close(cached_y, replay_targets[:ncheck], rtol=0, atol=0)
                replay_patches, replay_grid = model.backbone(replay_images.to(self.device))
                if tuple(replay_grid) != tuple(grid):
                    raise RuntimeError("Disk cache grid mismatch.")
                expected_features = _cache_features(replay_patches, task)[:ncheck].contiguous()
                _validate_cache_tensor(
                    cached_x, expected_features, label="disk_features_matched_batch",
                    report=cache_validation, atol=1e-5)
                _validate_cache_tensor(
                    cached_model(cached_x, task), cached_model(expected_features, task),
                    label="disk_head_matched_batch", report=cache_validation)
        finally:
            write_json(directory / "cache_validation.json", cache_validation)
        del cached_x, cached_y, stored, replay_images, replay_targets
        del replay_patches, expected_features, replay_loader
        metric = "top1_pct" if task == "classification" else "miou_pct"
        best_score, best_epoch = -float("inf"), None
        history = []
        total_steps = total_images = 0
        fit_start = time.perf_counter()
        for epoch in range(1, c["EPOCHS"][task] + 1):
            seed_everything(c["SEED"] + epoch)
            # Explicitly seed sampler and worker augmentation for each epoch.
            loaders["train"].generator.manual_seed(c["SEED"] + epoch)
            cached_model.train()
            if model.backbone.training:
                raise RuntimeError("Cached-head backbone must remain in evaluation mode.")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss_total = count_total = epoch_images = epoch_steps = 0
            progress = tqdm(loaders["train"], desc=f"{task} | {name} | epoch {epoch}/{c['EPOCHS'][task]}")
            for images, target in progress:
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with self.training_autocast():
                    logits = cached_model(images, task)
                loss_sum, count = loss_sum_and_count(logits.float(), target, task)
                if count:
                    loss = loss_sum / count
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss.")
                    loss.backward()
                    _check_gradients(model, list(head.parameters()), c["MAX_GRAD_NORM"])
                    optimizer.step()
                    epoch_steps += 1
                loss_total += loss_sum.detach().item()
                count_total += count
                epoch_images += images.shape[0]
                if epoch_steps and epoch_steps % 100 == 0:
                    progress.set_postfix(loss=loss_total / max(count_total, 1))
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - start
            peak_allocated = torch.cuda.max_memory_allocated() / 2**30
            peak_reserved = torch.cuda.max_memory_reserved() / 2**30
            total_steps += epoch_steps
            total_images += epoch_images
            if count_total == 0:
                raise RuntimeError("No valid training targets.")
            dev_start = time.perf_counter()
            dev = self.evaluate(cached_model, loaders["dev"], task)
            dev_seconds = time.perf_counter() - dev_start
            row = {
                "training_mode": "cached_head", "train_precision": c["TRAIN_PRECISION"], "evaluation_precision": "fp32", "evaluation_tf32": c.get("EVAL_TF32", False), "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                "train_loss": loss_total / count_total, "train_seconds": train_seconds,
                "train_images_per_second": epoch_images / train_seconds,
                "train_images": epoch_images, "optimizer_steps": epoch_steps,
                "train_peak_allocated_gib": peak_allocated, "train_peak_reserved_gib": peak_reserved,
                "dev_seconds": dev_seconds, **{f"dev_{k}": v for k, v in dev.items()},
            }
            history.append(row)
            write_csv(directory / "epochs.csv", history)
            checkpoint = {
                "model_name": name, "task": task, "epoch": epoch,
                "feature_dim": model.backbone.feature_dim, "input_size": c["SIZE"],
                "head_state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "dev_metrics": dev, "training_mode": "cached_head", "train_precision": c["TRAIN_PRECISION"], "evaluation_precision": "fp32", "evaluation_tf32": c.get("EVAL_TF32", False),
            }
            atomic_torch_save(checkpoint, directory / "last_head.pt")
            if dev[metric] > best_score:
                best_score, best_epoch = dev[metric], epoch
                atomic_torch_save(checkpoint, directory / "best_head.pt")
            scheduler.step()
            print(f"  train loss={row['train_loss']:.4f} | dev {metric}={dev[metric]:.2f} | train={train_seconds / 60:.1f} min | peak allocated={peak_allocated:.2f} GiB")
        fit_seconds = time.perf_counter() - fit_start
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scheduler, logits, loss_sum, images, target
        if "loss" in locals():
            del loss
        head.load_state_dict(torch.load(directory / "best_head.pt", map_location="cpu", weights_only=True)["head_state_dict"])
        test_start = time.perf_counter()
        heldout = self.evaluate(model, loaders["val"], task)
        test_seconds = time.perf_counter() - test_start
        images, target = check_images.to(self.device), check_target.to(self.device)
        if images.shape[0] != c["BATCH_SIZE"]:
            raise ValueError("Dev split must contain at least one full benchmark batch.")
        gc.collect()
        torch.cuda.empty_cache()
        speed = self.benchmark(model, images, task, directory)
        # Separate from GPU timing: operator counters can alter dispatch and
        # add overhead, so benchmark() above runs without these hooks.
        compute = {}
        try:
            compute.update(measure_flops_on_dataset(
                model, raw_dev_dataset, task, c["BATCH_SIZE"],
                c["FLOP_BATCHES"], c["SEED"], directory,
                module_names=(c["ATTENTION_MODULE_NAMES"] or {}).get(name),
            ))
            whole = compute["whole_model_forward_gflops_per_image"]
            if whole is None:
                print("Whole-model FLOPs require review:", compute["whole_model_flops_unmapped_ops"])
            else:
                print(f"Whole-model forward: {whole:.4f} GFLOPs/image (2 FLOPs/MAC).")
            value = compute["attention_forward_gflops_per_image"]
            if value is None:
                print("Attention FLOPs require coverage review:",
                      compute["attention_flops_unmapped_ops"],
                      compute["attention_flops_uncounted_modules"])
            else:
                print(f"Attention forward: {value:.4f} GFLOPs/image (2 FLOPs/MAC).")
        except Exception as exc:
            compute.update(whole_model_flops_coverage_ok=False,
                           whole_model_forward_gflops_per_image=None,
                           whole_model_flops_error=str(exc),
                           attention_flops_coverage_ok=False,
                           attention_forward_gflops_per_image=None,
                           attention_flops_error=str(exc))
            (directory / "attention_flops_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            print("Attention FLOP counting unavailable:", exc)
        try:
            compute.update(self.profile_model(model, images, target, task, directory))
            # Keep the new attention summary in the existing profiling file too.
            profiling = json.loads((directory / "profiling.json").read_text(encoding="utf-8"))
            profiling.update(compute)
            write_json(directory / "profiling.json", profiling)
        except Exception as exc:
            compute["profiling_error"] = str(exc)
            (directory / "profiling_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print("Profiling unavailable:", exc)
        result = {
            "model": name, "task": task, "status": "completed", "seed": c["SEED"],
            "resolution": c["SIZE"], "batch_size": c["BATCH_SIZE"], "epochs": c["EPOCHS"][task],
            "best_epoch": best_epoch, "best_dev_metric": best_score, "selection_metric": metric,
            "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
            "trainable_head_parameters": sum(p.numel() for p in head.parameters()),
            "total_training_images": total_images, "total_optimizer_steps": total_steps,
            "training_mode": "cached_head", "train_precision": c["TRAIN_PRECISION"], "evaluation_precision": "fp32", "evaluation_tf32": c.get("EVAL_TF32", False),
            "cache_dtype": "float32",
            "cache_train_seconds": cache_stats["train"]["seconds"],
            "cache_dev_seconds": cache_stats["dev"]["seconds"],
            "cache_build_seconds": sum(v["seconds"] for v in cache_stats.values()),
            "cache_bytes": sum(v["bytes"] for v in cache_stats.values()),
            "cache_backbone_image_forwards": sum(v["backbone_image_forwards"] for v in cache_stats.values()),
            "head_training_seconds": sum(r["train_seconds"] for r in history),
            "training_seconds": cache_stats["train"]["seconds"] + sum(r["train_seconds"] for r in history),
            "training_seconds_scope": "train feature extraction + cached head epochs; excludes dev",
            "pipeline_fit_seconds_including_cache_dev_and_saves": sum(v["seconds"] for v in cache_stats.values()) + fit_seconds,
            "experiment_seconds_including_measurements": time.perf_counter() - experiment_start,
            "profile_training_scope": "uncached representative forward + head backward; excludes optimizer",
            "inference_scope": "full original RGB model; no feature cache",
            "cache_directory": str(cache_dir),
            "fit_seconds_including_dev_and_saves": fit_seconds,
            "heldout_evaluation_seconds": test_seconds,
            "train_peak_allocated_gib": max(cache_stats["train"]["peak_allocated_gib"], max(r["train_peak_allocated_gib"] for r in history)),
            "train_peak_reserved_gib": max(cache_stats["train"]["peak_reserved_gib"], max(r["train_peak_reserved_gib"] for r in history)),
            **{f"heldout_{k}": v for k, v in heldout.items()}, **speed, **compute,
            "best_head_path": str(directory / "best_head.pt"),
        }
        # Close mappings explicitly before deleting files on Windows.
        for part in ("train", "dev"):
            loaders[part].dataset.close()
        result["cache_retained"] = True
        if not c["KEEP_CACHE"]:
            try:
                shutil.rmtree(cache_dir)
                result["cache_retained"] = False
            except OSError as exc:
                result["cache_cleanup_error"] = str(exc)
                print("Cache cleanup failed; training results are preserved:", exc)
        write_json(directory / "result.json", result)
        return result

    def run(self):
        self.prepare()
        results = []
        for task in self.c["TASKS"]:
            for name in self.c["MODEL_NAMES"]:
                print(f"\n{'=' * 65}\nStarting {task}: {name}\n{'=' * 65}")
                try:
                    result = self.run_experiment(name, task)
                except Exception as exc:
                    result = {"model": name, "task": task, "status": "failed", "error": str(exc)}
                    (self.out / f"{task}_{name}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        print("CUDA OOM: reduce MICRO_BATCH_SIZE for training, EVAL_BATCH_SIZE for evaluation, "
                              "or BATCH_SIZE for FP32 benchmarks.  GRAD_ACCUM_STEPS preserves the desired "
                              "effective training batch.  No automatic retry or batch change was performed.")
                    print("FAILED:", exc)
                    print("Full traceback:", self.out / f"{task}_{name}_error.txt")
                    if self.c.get("FAIL_FAST", True):
                        write_csv(self.out / "baseline measurments.csv", results + [result])
                        raise
                results.append(result)
                write_csv(self.out / "baseline measurments.csv", results)
                gc.collect()
                torch.cuda.empty_cache()
        print("\nFinished.  Measurements saved to:")
        print(self.out / "baseline measurments.csv")
        for result in results:
            print(result["task"], result["model"], "→", result["status"])
        return results, self.out


def run_baselines(
    model_factory, *, ROOT=None, SEED=42, SIZE=448,
    MODEL_NAMES=("dinov2", "siglip2", "radio"), EPOCHS=None,
    BATCH_SIZE=2, MICRO_BATCH_SIZE=1, GRAD_ACCUM_STEPS=16, EVAL_BATCH_SIZE=1, LR=None, WEIGHT_DECAY=0.01, NUM_WORKERS=0,
    BENCH_WARMUP=10, BENCH_REPEATS=50, PROFILE_REPEATS=3,
    IMAGENET_ROOT=None, ADE_ROOT=None, SPLIT_ROOT=None,
    ATTENTION_MODULE_NAMES=None, FLOP_BATCHES=3, BENCH_TRIALS=3,
    CACHE_ROOT=None, CACHE_BATCH_SIZE=None, KEEP_CACHE=False,
    TASKS=("classification", "segmentation"),
    TRAINING_MODE="lora", LORA_R=8, LORA_ALPHA=16.0, LORA_LR=1e-4,
    LORA_TARGETS=None, LORA_UNWRAP_BACKBONE_FORWARD=True,
    MAX_GRAD_NORM=1.0, STRICT_DETERMINISM=False,
    TRAIN_PRECISION="bf16", LOADER_BACKEND="process", FAIL_FAST=True,
    COMPILE_LORA=False, COMPILE_BACKEND="inductor", COMPILE_MODE="default", EVAL_TF32=False,
):
    """Run selected TASKS and MODEL_NAMES; return (results, OUT).

    Pass the existing BaselineModel class as model_factory.  It is constructed
    only in the parent process.  When called from a .py script (not a notebook),
    put the call under `if __name__ == "__main__":`.
    """
    ROOT = Path.cwd() if ROOT is None else Path(ROOT).expanduser().resolve()
    EPOCHS = dict(EPOCHS) if EPOCHS is not None else {"classification": 3, "segmentation": 5}
    LR = dict(LR) if LR is not None else {"classification": 1e-3, "segmentation": 1e-3}
    IMAGENET_ROOT = Path(IMAGENET_ROOT) if IMAGENET_ROOT is not None else ROOT / "datasets/imagenet"
    ADE_ROOT = Path(ADE_ROOT) if ADE_ROOT is not None else ROOT / "datasets/ade20k/ADEChallengeData2016"
    SPLIT_ROOT = Path(SPLIT_ROOT) if SPLIT_ROOT is not None else ROOT / "datasets/splits"
    MODEL_NAMES = tuple(MODEL_NAMES)
    TASKS = tuple(TASKS)
    if not TASKS or len(set(TASKS)) != len(TASKS) or any(
            t not in ("classification", "segmentation") for t in TASKS):
        raise ValueError("TASKS must contain classification and/or segmentation, without duplicates.")
    ATTENTION_MODULE_NAMES = dict(ATTENTION_MODULE_NAMES or {})
    CACHE_ROOT = (ROOT / "baseline feature cache" if CACHE_ROOT is None
                  else Path(CACHE_ROOT).expanduser().resolve())
    CACHE_BATCH_SIZE = BATCH_SIZE if CACHE_BATCH_SIZE is None else CACHE_BATCH_SIZE
    if not isinstance(KEEP_CACHE, bool):
        raise ValueError("KEEP_CACHE must be True or False.")
    if TRAINING_MODE not in ("cached_head", "lora"):
        raise ValueError("TRAINING_MODE must be cached_head or lora.")
    if TRAIN_PRECISION not in ("fp32", "bf16"):
        raise ValueError("TRAIN_PRECISION must be fp32 or bf16.")
    if not isinstance(COMPILE_LORA, bool) or not isinstance(EVAL_TF32, bool):
        raise ValueError("COMPILE_LORA and EVAL_TF32 must be booleans.")
    if COMPILE_LORA and TRAINING_MODE != "lora":
        raise ValueError("COMPILE_LORA requires TRAINING_MODE='lora'.")
    if not isinstance(COMPILE_BACKEND, str) or not COMPILE_BACKEND:
        raise ValueError("COMPILE_BACKEND must be a nonempty backend name.")
    if COMPILE_MODE not in ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
        raise ValueError("Unsupported COMPILE_MODE.")
    if LOADER_BACKEND not in ("process", "thread"):
        raise ValueError("LOADER_BACKEND must be process or thread.")
    if not isinstance(FAIL_FAST, bool):
        raise ValueError("FAIL_FAST must be boolean.")
    LORA_TARGETS = dict(LORA_TARGETS or {})
    if any(not isinstance(v, (list, tuple)) or not all(isinstance(n, str) for n in v)
           for v in LORA_TARGETS.values()):
        raise ValueError("LORA_TARGETS must map model names to lists of exact projection paths.")
    if isinstance(LORA_R, bool) or not isinstance(LORA_R, int) or LORA_R < 1:
        raise ValueError("LORA_R must be a positive integer.")
    for value in (LORA_ALPHA, LORA_LR):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("LoRA alpha and learning rate must be positive and finite.")
    if math.isnan(MAX_GRAD_NORM) or MAX_GRAD_NORM <= 0:
        raise ValueError("MAX_GRAD_NORM must be positive (infinity disables clipping).")
    if not math.isfinite(WEIGHT_DECAY) or WEIGHT_DECAY < 0:
        raise ValueError("WEIGHT_DECAY must be finite and nonnegative.")
    if not isinstance(STRICT_DETERMINISM, bool) or not isinstance(LORA_UNWRAP_BACKBONE_FORWARD, bool):
        raise ValueError("Determinism and forward unwrap options must be booleans.")
    config = dict(locals())
    config.pop("model_factory")
    if not callable(model_factory):
        raise TypeError("Pass your BaselineModel class as the first argument.")
    for key in ("SIZE", "BATCH_SIZE", "BENCH_REPEATS", "PROFILE_REPEATS", "FLOP_BATCHES", "BENCH_TRIALS", "CACHE_BATCH_SIZE", "MICRO_BATCH_SIZE", "GRAD_ACCUM_STEPS", "EVAL_BATCH_SIZE"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("NUM_WORKERS", "BENCH_WARMUP"):
        if not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    for task in TASKS:
        if (not isinstance(EPOCHS[task], int) or EPOCHS[task] < 1
                or not math.isfinite(LR[task]) or LR[task] <= 0):
            raise ValueError(f"{task} requires positive epochs and learning rate.")
    if not MODEL_NAMES or len(set(MODEL_NAMES)) != len(MODEL_NAMES):
        raise ValueError("MODEL_NAMES must be nonempty with no duplicates.")
    return _Runner(model_factory, config).run()


def add_forward_flops_to_run(model_factory, run_dir, *, ROOT=None,
                               IMAGENET_ROOT=None, ADE_ROOT=None, SPLIT_ROOT=None,
                               ATTENTION_MODULE_NAMES=None, FLOP_BATCHES=3,
                               REBENCHMARK=False, BENCH_TRIALS=3):
    """Backfill consistent forward FLOPs for completed experiments, no retraining.

    Loads saved best heads and their original local backbones. Updates result.json,
    profiling.json and CSV with counts; detailed audits go in flop_samples/.
    Optional REBENCHMARK=True reruns inference timing using the recorded warmup,
    repeats and batch size, and BENCH_TRIALS independent trials. It preserves the
    first pre-refresh timing fields in benchmark_before_refresh.json.
    Checkpoints, training history and accuracy metrics remain unchanged.
    """
    for key, value in [('FLOP_BATCHES', FLOP_BATCHES), ('BENCH_TRIALS', BENCH_TRIALS)]:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.  Select your CUDA PyTorch notebook kernel.')
    run_dir = Path(run_dir).expanduser().resolve()
    root = Path.cwd() if ROOT is None else Path(ROOT).expanduser().resolve()
    config = json.loads((run_dir / 'configuration.json').read_text(encoding='utf-8'))
    roots = {
        'classification': Path(IMAGENET_ROOT) if IMAGENET_ROOT is not None else root / 'datasets/imagenet',
        'segmentation': Path(ADE_ROOT) if ADE_ROOT is not None else root / 'datasets/ade20k/ADEChallengeData2016',
    }
    split_root = Path(SPLIT_ROOT) if SPLIT_ROOT is not None else root / 'datasets/splits'
    size, batch_size = config['input_size'], config['batch_size']
    names_by_model = (config.get('attention_module_names') or {}) if ATTENTION_MODULE_NAMES is None else ATTENTION_MODULE_NAMES
    csv_path = run_dir / 'baseline measurments.csv'
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.cuda.set_device(0)
    _set_tf32(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for row in rows:
        if row.get('status') != 'completed':
            continue
        name, task = row['model'], row['task']
        directory = run_dir / f'{task}_{name}'
        result_path = directory / 'result.json'
        # Read canonical numeric results rather than round-tripping them through CSV strings.
        result = json.loads(result_path.read_text(encoding='utf-8'))
        split_file = split_root / ('imagenet_split.json' if task == 'classification' else 'ade20k_split.json')
        expected_hash = config.get('split_sha256', {}).get(task)
        if expected_hash and hashlib.sha256(split_file.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'Saved split has changed since training: {split_file}')
        split = json.loads(split_file.read_text(encoding='utf-8'))
        ds = SavedSplitDataset(task, 'dev', split['dev_files'], roots[task], size,
                               split.get('class_to_idx', {}), {})
        if len(ds) < batch_size:
            raise ValueError(f'Dev split is smaller than the recorded batch size: {task}')
        model = images = None
        print(f'Measuring forward FLOPs: {task} / {name}')
        try:
            seed_everything(config['seed'])
            model = model_factory(name)
            inactive = 'segmentation' if task == 'classification' else 'classification'
            setattr(model, f'{inactive}_head', nn.Identity())
            checkpoint = torch.load(directory / ('best_lora.pt' if config.get('training_mode') == 'lora' else 'best_head.pt'), map_location='cpu', weights_only=True)
            if (checkpoint['model_name'] != name or checkpoint['task'] != task or
                    checkpoint['input_size'] != size):
                raise ValueError('Checkpoint metadata does not match this experiment.')
            load_baseline_checkpoint(model, checkpoint, merge=True)
            model = model.to('cuda:0').eval()
            if REBENCHMARK:
                backup = directory / 'benchmark_before_refresh.json'
                if not backup.exists():
                    write_json(backup, {k: v for k, v in result.items()
                                       if k.startswith(('b1_', f'b{batch_size}_', 'benchmark_'))})
                timing_runner = _Runner(model_factory, {
                    'ROOT': root, 'IMAGENET_ROOT': roots['classification'],
                    'ADE_ROOT': roots['segmentation'], 'SPLIT_ROOT': split_root,
                    'BATCH_SIZE': batch_size, 'BENCH_TRIALS': BENCH_TRIALS,
                    'BENCH_WARMUP': config.get('bench_warmup', 10),
                    'BENCH_REPEATS': config.get('bench_repeats', 50),
                })
                images = torch.stack([ds[i][0] for i in range(batch_size)]).to('cuda:0')
                result.update(timing_runner.benchmark(model, images, task, directory))
                del images
                images = None
            summary = measure_flops_on_dataset(
                model, ds, task, batch_size, FLOP_BATCHES, config['seed'], directory,
                module_names=names_by_model.get(name))
            # Clear stale attention fields if an earlier backfill had failed.
            for key in list(result):
                if key.startswith(('attention_', 'whole_model_', 'flops_sample_')) and key != 'attention_gpu_time_pct':
                    result.pop(key)
            result.update(summary)
            write_json(result_path, result)
            profile_path = directory / 'profiling.json'
            if profile_path.exists():
                profiling = json.loads(profile_path.read_text(encoding='utf-8'))
                for key in list(profiling):
                    if key.startswith(('attention_', 'whole_model_', 'flops_sample_')) and key != 'attention_gpu_time_pct':
                        profiling.pop(key)
                profiling.update(summary)
                write_json(profile_path, profiling)
            row.clear()
            row.update(result)
            write_csv(csv_path, rows)
            write_json(directory / 'measurement_refresh.json', {
                'timestamp': datetime.now().isoformat(), 'counter_version': 3,
                'flop_batches': FLOP_BATCHES, 'timing_refreshed': REBENCHMARK,
                'benchmark_trials': BENCH_TRIALS if REBENCHMARK else result.get('benchmark_trials'),
                'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                'precision': 'float32; TF32 disabled',
            })
            value = summary['whole_model_forward_gflops_per_image']
            if value is not None:
                print(f'  {value:.4f} GFLOPs/image (2 FLOPs/MAC).')
            else:
                print('  Coverage review needed; inspect forward_flops.json and flop_samples/.')
        finally:
            del model, images
            gc.collect()
            torch.cuda.empty_cache()
    return rows, run_dir


def add_attention_flops_to_run(*args, **kwargs):
    """Compatibility alias; backfills both whole-model and attention FLOPs."""
    return add_forward_flops_to_run(*args, **kwargs)


def refresh_component_timings(model_factory, run_dir, *, ROOT=None,
                              MODEL_NAMES=("radio",), TASKS=("segmentation",),
                              PROFILE_REPEATS=10, WARMUP=10):
    """Refresh forward component attribution from saved heads, without training.

    Preserves old reports in a timestamped backup. Updates profiling.json,
    result.json and the run CSV; accuracy, FLOPs and benchmark speed stay intact.
    """
    if PROFILE_REPEATS < 1 or WARMUP < 0:
        raise ValueError("Require positive repeats and nonnegative warmup.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    run_dir = Path(run_dir).expanduser().resolve()
    root = Path.cwd() if ROOT is None else Path(ROOT).expanduser().resolve()
    config = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
    csv_path = run_dir / "baseline measurments.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected = [r for r in rows if r.get("status") == "completed"
                and r["model"] in MODEL_NAMES and r["task"] in TASKS]
    if not selected:
        raise ValueError("No completed experiments match the selected models/tasks in this folder.")
    backup = run_dir / ("component_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    backup.mkdir()
    shutil.copy2(csv_path, backup / csv_path.name)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.cuda.set_device(0)
    _set_tf32(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    size, batch = config["input_size"], config["batch_size"]
    for row in selected:
        name, task = row["model"], row["task"]
        directory = run_dir / f"{task}_{name}"
        split_path = root / "datasets/splits" / ("imagenet_split.json" if task == "classification" else "ade20k_split.json")
        expected = config.get("split_sha256", {}).get(task)
        if expected and hashlib.sha256(split_path.read_bytes()).hexdigest() != expected:
            raise ValueError("Dataset split changed since training.")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        data_root = root / ("datasets/imagenet" if task == "classification" else "datasets/ade20k/ADEChallengeData2016")
        ds = SavedSplitDataset(task, "dev", split["dev_files"], data_root,
                               size, split.get("class_to_idx", {}), {})
        if len(ds) < batch:
            raise ValueError("Dev split smaller than recorded batch size.")
        saved = backup / directory.name
        saved.mkdir()
        for filename in ("result.json", "profiling.json"):
            if (directory / filename).exists():
                shutil.copy2(directory / filename, saved / filename)
        model = images = None
        try:
            seed_everything(config["seed"])
            model = model_factory(name)
            inactive = "segmentation" if task == "classification" else "classification"
            setattr(model, f"{inactive}_head", nn.Identity())
            checkpoint = torch.load(directory / ("best_lora.pt" if config.get("training_mode") == "lora" else "best_head.pt"), map_location="cpu", weights_only=True)
            if (checkpoint["model_name"], checkpoint["task"], checkpoint["input_size"]) != (name, task, size):
                raise ValueError("Checkpoint metadata mismatch.")
            load_baseline_checkpoint(model, checkpoint, merge=True)
            model = model.to("cuda:0").eval()
            images = torch.stack([ds[i][0] for i in range(batch)]).to("cuda:0")
            with torch.no_grad():
                for _ in range(WARMUP):
                    model(images, task)
                torch.cuda.synchronize()
                with component_labels(model, task) as matched:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        for _ in range(PROFILE_REPEATS):
                            with record_function("component/total"):
                                model(images, task)
                        torch.cuda.synchronize()
            prof.export_chrome_trace(str(directory / "component_refresh_trace.json"))
            summary = _component_time_summary(prof, matched)
            summary.update(component_profile_repeats=PROFILE_REPEATS,
                           component_profile_batch_size=batch)
            for filename in ("profiling.json", "result.json"):
                path = directory / filename
                doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                doc.update(summary)
                if filename == "profiling.json":
                    doc["matched_modules"] = matched
                write_json(path, doc)
                if filename == "result.json":
                    row.clear()
                    row.update(doc)
            write_csv(csv_path, rows)
            print(name, task, "available:", summary["component_gpu_attribution_available"],
                  summary["component_gpu_attribution_reason"])
        finally:
            del model, images
            gc.collect()
            torch.cuda.empty_cache()
    return rows, run_dir
