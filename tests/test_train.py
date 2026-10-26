import torch
from torch import nn

from catdog.model import build_model, load_model
from catdog.train import TrainConfig, train, train_step


def test_train_steps_lower_the_loss_on_a_toy_batch():
    torch.manual_seed(0)
    model = build_model(pretrained=False)
    images = torch.randn(4, 3, 64, 64)
    labels = torch.tensor([0, 1, 0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = [train_step(model, images, labels, optimizer, nn.CrossEntropyLoss()) for _ in range(5)]
    assert losses[-1] < losses[0]


def test_train_end_to_end_on_tiny_dataset(fake_pets, tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    cfg = TrainConfig(
        data_dir=fake_pets,
        checkpoint=checkpoint,
        epochs=2,
        batch_size=2,
        val_fraction=0.34,
        device="cpu",
        num_workers=0,
        pretrained=False,
    )
    meta = train(cfg)
    assert [row["epoch"] for row in meta["history"]] == [1, 2]
    assert all(0 <= row["val_accuracy"] <= 1 for row in meta["history"])
    assert meta["config"]["checkpoint"] == str(checkpoint)
    assert isinstance(load_model(checkpoint, torch.device("cpu")), nn.Module)
