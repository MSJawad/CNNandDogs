"""Command line interface: catdog {download,train,evaluate,predict}."""

import argparse
import sys
from pathlib import Path

from catdog.train import TrainConfig

DEFAULTS = TrainConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="catdog", description="Cat vs dog image classifier (ResNet-18 on Oxford-IIIT Pet)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, data: bool = True) -> None:
        if data:
            p.add_argument("--data-dir", type=Path, default=DEFAULTS.data_dir)
        p.add_argument("--checkpoint", type=Path, default=DEFAULTS.checkpoint)
        p.add_argument("--device", default=DEFAULTS.device, help="auto, cpu, mps or cuda")

    p = sub.add_parser("download", help="download Oxford-IIIT Pet (about 800 MB)")
    p.add_argument("--data-dir", type=Path, default=DEFAULTS.data_dir)

    p = sub.add_parser("train", help="fine-tune ResNet-18 and save a checkpoint")
    add_common(p)
    p.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    p.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    p.add_argument("--lr", type=float, default=DEFAULTS.lr)
    p.add_argument("--seed", type=int, default=DEFAULTS.seed)
    p.add_argument("--num-workers", type=int, default=DEFAULTS.num_workers)

    p = sub.add_parser("evaluate", help="score a checkpoint on the held-out test split")
    add_common(p)
    p.add_argument("--out", type=Path, default=Path("docs"), help="where results go")
    p.add_argument("--num-workers", type=int, default=DEFAULTS.num_workers)

    p = sub.add_parser("predict", help="classify image files as cat or dog")
    add_common(p, data=False)
    p.add_argument("images", type=Path, nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "download":
        from catdog.data import download

        print(f"downloaded to {download(args.data_dir)}")
    elif args.command == "train":
        from catdog.train import train

        train(
            TrainConfig(
                data_dir=args.data_dir,
                checkpoint=args.checkpoint,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=args.device,
                num_workers=args.num_workers,
            )
        )
    elif args.command == "evaluate":
        from catdog.evaluate import evaluate

        r = evaluate(
            args.checkpoint, args.data_dir, args.out, args.device, num_workers=args.num_workers
        )
        print(
            f"test n={r['n']}  accuracy={r['accuracy']:.4f}  balanced={r['balanced_accuracy']:.4f}"
        )
        for name, c in r["per_class"].items():
            print(f"  {name}: precision={c['precision']:.4f} recall={c['recall']:.4f}")
        print(f"wrote results.json, confusion_matrix.png and examples.jpg to {args.out}")
    elif args.command == "predict":
        from catdog.model import load_model, pick_device
        from catdog.predict import format_prob, predict_images

        missing = [p for p in args.images if not p.is_file()]
        if missing:
            print(f"error: no such file: {', '.join(map(str, missing))}", file=sys.stderr)
            return 2
        device = pick_device(args.device)
        try:
            model = load_model(args.checkpoint, device)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        try:
            predictions = predict_images(model, args.images, device)
        except OSError as e:
            print(f"error: could not read image: {e}", file=sys.stderr)
            return 2
        for path, (label, prob) in zip(args.images, predictions, strict=True):
            print(f"{path}: {label} ({format_prob(prob)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
