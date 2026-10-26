import json
import re

import numpy as np
import pytest
import torch
from conftest import write_image

from catdog.cli import main
from catdog.evaluate import pick_examples
from catdog.model import build_model, save_checkpoint
from catdog.predict import format_prob


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "model.pt"
    save_checkpoint(build_model(pretrained=False), path, {})
    return path


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("download", "train", "evaluate", "predict"):
        assert command in out


def test_predict_prints_a_label_per_image(tmp_path, checkpoint, capsys):
    images = [write_image(tmp_path / f"img{i}.png", (320, 200), seed=i) for i in range(2)]
    args = ["predict", "--checkpoint", str(checkpoint), "--device", "cpu", *map(str, images)]
    assert main(args) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    for image, line in zip(images, lines, strict=True):
        assert re.fullmatch(rf"{re.escape(str(image))}: (cat|dog) \((>99\.9|\d+\.\d)%\)", line)


def test_predict_missing_checkpoint(tmp_path, capsys):
    image = write_image(tmp_path / "img.png", (64, 64), seed=0)
    assert main(["predict", "--checkpoint", str(tmp_path / "none.pt"), str(image)]) == 2
    assert "catdog train" in capsys.readouterr().err


def test_predict_missing_image(tmp_path, checkpoint, capsys):
    assert main(["predict", "--checkpoint", str(checkpoint), str(tmp_path / "nope.jpg")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_predict_unreadable_image(tmp_path, checkpoint, capsys):
    bad = tmp_path / "notes.jpg"
    bad.write_text("not an image")
    assert main(["predict", "--checkpoint", str(checkpoint), "--device", "cpu", str(bad)]) == 2
    assert "could not read image" in capsys.readouterr().err


def test_evaluate_writes_results_and_figures(fake_pets, checkpoint, tmp_path, capsys):
    out = tmp_path / "results"
    args = ["evaluate", "--data-dir", str(fake_pets), "--checkpoint", str(checkpoint)]
    assert main([*args, "--out", str(out), "--device", "cpu", "--num-workers", "0"]) == 0
    result = json.loads((out / "results.json").read_text())
    assert result["n"] == 6
    assert sum(map(sum, result["confusion_matrix"])) == 6
    assert (out / "confusion_matrix.png").stat().st_size > 0
    assert (out / "examples.jpg").stat().st_size > 0
    assert "accuracy=" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("p", "text"), [(0.5, "50.0%"), (0.987, "98.7%"), (0.999, "99.9%"), (0.9996, ">99.9%")]
)
def test_format_prob_never_shows_100_percent(p, text):
    assert format_prob(p) == text


def test_pick_examples_puts_most_confident_mistakes_last():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4], [0.3, 0.7], [0.05, 0.95]])
    picked = pick_examples(y_true, probs, n=4, seed=0)
    assert len(picked) == 4
    assert picked[-2:] == [5, 1]
    assert set(picked[:2]) <= {0, 2, 4}


def test_train_without_dataset_points_to_download(tmp_path, capsys):
    checkpoint = tmp_path / "model.pt"
    args = ["train", "--data-dir", str(tmp_path / "nodata"), "--checkpoint", str(checkpoint)]
    assert main([*args, "--device", "cpu", "--num-workers", "0"]) == 2
    assert "catdog download" in capsys.readouterr().err
    assert not checkpoint.exists()


def test_evaluate_without_dataset_points_to_download(tmp_path, checkpoint, capsys):
    args = ["evaluate", "--data-dir", str(tmp_path / "nodata"), "--checkpoint", str(checkpoint)]
    assert main([*args, "--out", str(tmp_path / "out"), "--device", "cpu"]) == 2
    assert "catdog download" in capsys.readouterr().err


def test_evaluate_missing_checkpoint_points_to_train(fake_pets, tmp_path, capsys):
    args = ["evaluate", "--data-dir", str(fake_pets), "--checkpoint", str(tmp_path / "none.pt")]
    assert main([*args, "--out", str(tmp_path / "out"), "--device", "cpu"]) == 2
    assert "catdog train" in capsys.readouterr().err
