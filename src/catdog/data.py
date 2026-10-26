"""Oxford-IIIT Pet loading, cat/dog labels, train/validation split and image transforms."""

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

CLASSES = ("cat", "dog")
IMAGE_SIZE = 224
# ImageNet statistics, matching the pretrained backbone.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def train_transform() -> Callable:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def display_transform() -> Callable:
    """Deterministic resize and center crop, still a PIL image."""
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)])


def eval_transform() -> Callable:
    return transforms.Compose(
        [display_transform(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


def download(data_dir: Path) -> Path:
    """Download and extract Oxford-IIIT Pet into data_dir, then delete the archives."""
    OxfordIIITPet(data_dir, split="trainval", download=True)
    base = Path(data_dir) / "oxford-iiit-pet"
    for archive in base.glob("*.tar.gz"):
        archive.unlink()
    return base


def load_split(data_dir: Path, split: str) -> OxfordIIITPet:
    """Load the official "trainval" or "test" split with cat=0, dog=1 labels.

    Items are (PIL image, label); wrap the result in TransformedSubset to get tensors.
    """
    return OxfordIIITPet(data_dir, split=split, target_types="binary-category")


def train_val_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Shuffle range(n) with a fixed seed and hold out val_fraction of it for validation."""
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = max(1, round(n * val_fraction))
    return sorted(perm[n_val:]), sorted(perm[:n_val])


class TransformedSubset(Dataset):
    """A view of selected items of a (PIL image, label) dataset with its own transform."""

    def __init__(self, dataset: Dataset, indices: Sequence[int], transform: Callable):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[self.indices[i]]
        return self.transform(image), label
