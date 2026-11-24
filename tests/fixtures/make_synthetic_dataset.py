"""Create a tiny ImageNet-like dataset for tests (no real data needed)."""

from pathlib import Path

from PIL import Image

from unlearning.classes import load_class_index


def make_dataset(
    root: Path,
    indices: tuple[int, ...],
    train_counts: dict[int, int],
    val_count: int = 6,
) -> Path:
    index = load_class_index()
    for position, index_0b in enumerate(indices):
        synset = index.get(index_0b).synset
        for split, count in (("train", train_counts[index_0b]), ("val", val_count)):
            directory = root / split / synset
            directory.mkdir(parents=True, exist_ok=True)
            for number in range(count):
                color = (10 * position, 20 * number, 30 * number)
                path = directory / f"{synset}_{split}_{number:03d}.JPEG"
                Image.new("RGB", (32, 32), color).save(path)
    return root
