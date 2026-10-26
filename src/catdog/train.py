"""Fine-tuning loop."""

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from catdog import data
from catdog.model import build_model, pick_device, save_checkpoint


@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    checkpoint: Path = Path("checkpoints/catdog-resnet18.pt")
    epochs: int = 3
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    seed: int = 0
    device: str = "auto"
    num_workers: int = 4
    pretrained: bool = True  # False only for offline tests


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int = 0
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def train_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(images), labels)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return (true labels, class probabilities) for every item in loader."""
    model.eval()
    labels, probs = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(targets)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


def train(cfg: TrainConfig) -> dict:
    """Fine-tune on the trainval split minus a validation hold-out; save the final model."""
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)

    trainval = data.load_split(cfg.data_dir, "trainval")
    train_idx, val_idx = data.train_val_indices(len(trainval), cfg.val_fraction, cfg.seed)
    train_ds = data.TransformedSubset(trainval, train_idx, data.train_transform())
    val_ds = data.TransformedSubset(trainval, val_idx, data.eval_transform())
    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, cfg.seed)
    val_loader = make_loader(val_ds, cfg.batch_size, False, cfg.num_workers)
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} epochs={cfg.epochs}")

    model = build_model(pretrained=cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * len(train_loader)
    )
    loss_fn = nn.CrossEntropyLoss()

    history = []
    start = time.monotonic()
    for epoch in range(1, cfg.epochs + 1):
        losses = []
        for images, labels in train_loader:
            losses.append(
                train_step(model, images.to(device), labels.to(device), optimizer, loss_fn)
            )
            scheduler.step()
        y_true, probs = predict_loader(model, val_loader, device)
        val_acc = float((probs.argmax(1) == y_true).mean())
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_accuracy": val_acc,
            "elapsed_s": round(time.monotonic() - start, 1),
        }
        history.append(row)
        print(
            f"epoch {epoch}/{cfg.epochs}  train_loss={row['train_loss']:.4f}  "
            f"val_acc={val_acc:.4f}  elapsed={row['elapsed_s']}s"
        )

    meta = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "device": str(device),
        "history": history,
    }
    save_checkpoint(model, cfg.checkpoint, meta)
    print(f"saved {cfg.checkpoint}")
    return meta
