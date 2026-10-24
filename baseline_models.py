"""Common backbone, preprocessing and task heads for both comparison reports."""
import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

def resolve_project_root(root=None):
    """Locate shared pretrained weights and datasets for both reports."""
    if root is not None:
        return Path(root).expanduser().resolve()
    for candidate in (Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent):
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

    # Autograd remains enabled for LoRA in both reports.
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


