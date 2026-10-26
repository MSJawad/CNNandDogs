"""Classify image files with a trained checkpoint."""

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from catdog.data import CLASSES, eval_transform


def format_prob(p: float) -> str:
    """Percentage with one decimal, without rounding near-certain predictions up to 100%."""
    return ">99.9%" if p > 0.999 else f"{p:.1%}"


@torch.no_grad()
def predict_images(
    model: nn.Module, paths: Sequence[Path], device: torch.device
) -> list[tuple[str, float]]:
    """Return (label, probability of that label) for each image file."""
    transform = eval_transform()
    batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths])
    probs = torch.softmax(model.eval()(batch.to(device)), dim=1).cpu()
    conf, idx = probs.max(dim=1)
    return [(CLASSES[i], c) for i, c in zip(idx.tolist(), conf.tolist(), strict=True)]
