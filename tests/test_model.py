import pytest
import torch

from catdog.model import build_model, load_model, pick_device, save_checkpoint


def test_forward_pass_gives_two_logits():
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(3, 3, 224, 224))
    assert logits.shape == (3, 2)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = build_model(pretrained=False).eval()
    path = tmp_path / "nested" / "model.pt"
    save_checkpoint(model, path, {"note": "test"})
    loaded = load_model(path, torch.device("cpu"))
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))
    assert not loaded.training


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="catdog train"):
        load_model(tmp_path / "missing.pt", torch.device("cpu"))


def test_pick_device():
    assert pick_device("cpu") == torch.device("cpu")
    assert pick_device("auto").type in {"cpu", "mps", "cuda"}
