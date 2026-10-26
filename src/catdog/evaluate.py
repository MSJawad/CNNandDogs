"""Test-set evaluation: metrics JSON, confusion matrix and example prediction figures."""

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from catdog import data, metrics
from catdog.model import load_model, pick_device
from catdog.predict import format_prob
from catdog.train import make_loader, predict_loader


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    labels = [c.capitalize() for c in data.CLASSES]
    row_frac = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig = Figure(figsize=(4.2, 3.6), dpi=150)
    ax = fig.subplots()
    ax.imshow(row_frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if row_frac[i, j] > 0.5 else "black"
            text = f"{cm[i, j]}\n{row_frac[i, j]:.1%}"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Oxford-IIIT Pet test set (n={cm.sum()})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def pick_examples(y_true: np.ndarray, probs: np.ndarray, n: int, seed: int) -> list[int]:
    """Up to two of the most confident mistakes, then random correct predictions to fill n."""
    pred = probs.argmax(1)
    wrong = np.flatnonzero(pred != y_true)
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:2]
    right = np.flatnonzero(pred == y_true)
    rng = np.random.default_rng(seed)
    right = rng.choice(right, size=min(n - len(wrong), len(right)), replace=False)
    return sorted(right.tolist()) + wrong.tolist()


def plot_examples(dataset, indices: list[int], y_true, probs, path: Path) -> None:
    cols = 4
    rows = -(-len(indices) // cols)
    fig = Figure(figsize=(cols * 2.4, rows * 2.75), dpi=150)
    axes = fig.subplots(rows, cols)
    to_display = data.display_transform()
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), indices, strict=False):
        image, _ = dataset[i]
        pred = int(probs[i].argmax())
        correct = pred == y_true[i]
        ax.imshow(to_display(image))
        ax.set_title(
            f"{data.CLASSES[pred]} {format_prob(probs[i, pred])}\n"
            f"{'correct' if correct else 'wrong, true: ' + data.CLASSES[y_true[i]]}",
            fontsize=9,
            color="#1a7f37" if correct else "#cf222e",
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pil_kwargs={"quality": 85})


def evaluate(
    checkpoint: Path,
    data_dir: Path,
    out_dir: Path,
    device: str = "auto",
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 0,
) -> dict:
    """Score the checkpoint on the official test split and write results to out_dir."""
    torch_device = pick_device(device)
    model = load_model(checkpoint, torch_device)
    test = data.load_split(data_dir, "test")
    loader = make_loader(
        data.TransformedSubset(test, range(len(test)), data.eval_transform()),
        batch_size,
        False,
        num_workers,
    )
    y_true, probs = predict_loader(model, loader, torch_device)
    cm = metrics.confusion_matrix(y_true, probs.argmax(1), len(data.CLASSES))
    result = {"split": "test", "checkpoint": str(checkpoint), **metrics.report(cm, data.CLASSES)}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    examples = pick_examples(y_true, probs, n=8, seed=seed)
    plot_examples(test, examples, y_true, probs, out_dir / "examples.jpg")
    return result
