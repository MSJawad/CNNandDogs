from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# (image id, 1-based breed id, species: 1 = cat, 2 = dog), in the Oxford-IIIT Pet list format.
FAKE_ITEMS = [
    ("Abyssinian_1", 1, 1),
    ("Abyssinian_2", 1, 1),
    ("Bengal_1", 6, 1),
    ("beagle_1", 4, 2),
    ("beagle_2", 4, 2),
    ("pug_1", 26, 2),
]


def write_image(path: Path, size: tuple[int, int], seed: int) -> Path:
    pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


@pytest.fixture
def fake_pets(tmp_path: Path) -> Path:
    """A tiny on-disk copy of the Oxford-IIIT Pet layout; returns the data dir."""
    base = tmp_path / "oxford-iiit-pet"
    (base / "images").mkdir(parents=True)
    (base / "annotations").mkdir()
    for split in ("trainval", "test"):
        lines = [f"{image_id} {breed} {species} 1" for image_id, breed, species in FAKE_ITEMS]
        (base / "annotations" / f"{split}.txt").write_text("\n".join(lines) + "\n")
    for i, (image_id, _, _) in enumerate(FAKE_ITEMS):
        write_image(base / "images" / f"{image_id}.jpg", (300 + 10 * i, 240), seed=i)
    return tmp_path
