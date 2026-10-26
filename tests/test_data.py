import pytest
import torch
from PIL import Image

from catdog import data


def test_load_split_maps_species_to_cat_and_dog(fake_pets):
    ds = data.load_split(fake_pets, "trainval")
    labels = [ds[i][1] for i in range(len(ds))]
    assert labels == [0, 0, 0, 1, 1, 1]
    assert isinstance(ds[0][0], Image.Image)


def test_transformed_subset_yields_normalized_tensors(fake_pets):
    ds = data.load_split(fake_pets, "test")
    for transform in (data.train_transform(), data.eval_transform()):
        subset = data.TransformedSubset(ds, [1, 4], transform)
        assert len(subset) == 2
        image, label = subset[1]
        assert image.shape == (3, data.IMAGE_SIZE, data.IMAGE_SIZE)
        assert image.dtype == torch.float32
        assert label == 1


def test_train_val_indices_are_disjoint_complete_and_seeded():
    train, val = data.train_val_indices(100, 0.1, seed=0)
    assert len(val) == 10
    assert sorted(train + val) == list(range(100))
    assert data.train_val_indices(100, 0.1, seed=0) == (train, val)
    assert data.train_val_indices(100, 0.1, seed=1) != (train, val)


@pytest.mark.parametrize("fraction", [0, 1, -0.1])
def test_train_val_indices_rejects_bad_fraction(fraction):
    with pytest.raises(ValueError):
        data.train_val_indices(10, fraction, seed=0)


def test_download_removes_archives(tmp_path, monkeypatch):
    base = tmp_path / "oxford-iiit-pet"

    def fake_dataset(root, split, download):
        base.mkdir()
        (base / "images.tar.gz").write_bytes(b"x")
        (base / "images").mkdir()

    monkeypatch.setattr(data, "OxfordIIITPet", fake_dataset)
    assert data.download(tmp_path) == base
    assert not list(base.glob("*.tar.gz"))
    assert (base / "images").is_dir()


def test_load_split_without_dataset_points_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="catdog download"):
        data.load_split(tmp_path, "trainval")
