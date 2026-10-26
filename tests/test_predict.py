import torch
from PIL import Image
from torch import nn

from catdog.predict import predict_images


class CountingModel(nn.Module):
    """Wraps a model and records the batch size of every forward pass."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(x.shape[0])
        return self.inner(x)


def red_detector() -> nn.Module:
    """Cat when the image is redder than the ImageNet mean, dog otherwise."""
    head = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), head)


def test_predict_images_runs_in_fixed_size_batches(tmp_path):
    colors = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
    paths = []
    for i, color in enumerate(colors):
        paths.append(tmp_path / f"img{i}.png")
        Image.new("RGB", (40 + 10 * i, 30), color).save(paths[-1])
    model = CountingModel(red_detector())
    device = torch.device("cpu")

    batched = predict_images(model, paths, device, batch_size=2)
    assert model.batch_sizes == [2, 2, 1]
    assert [label for label, _ in batched] == ["cat", "dog", "cat", "dog", "dog"]
    assert all(prob > 0.9 for _, prob in batched)

    model.batch_sizes.clear()
    single = predict_images(model, paths, device, batch_size=len(paths))
    assert model.batch_sizes == [5]
    assert [label for label, _ in single] == [label for label, _ in batched]
    single_probs = torch.tensor([prob for _, prob in single])
    batched_probs = torch.tensor([prob for _, prob in batched])
    assert torch.allclose(single_probs, batched_probs)
