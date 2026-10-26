"""ResNet-18 with a two-class head, checkpoint I/O and device selection."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from catdog.data import CLASSES


def build_model(pretrained: bool = True) -> nn.Module:
    """ResNet-18 (ImageNet weights when pretrained) with its classifier replaced by cat/dog."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model: nn.Module, path: Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {"arch": "resnet18", "classes": list(CLASSES), "meta": meta, "state_dict": state}, path
    )


def load_model(path: Path, device: torch.device) -> nn.Module:
    """Load a checkpoint written by save_checkpoint, in eval mode on device."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path} (train one with `catdog train`)")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()
