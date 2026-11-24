import pytest

from unlearning.classes import load_class_index
from unlearning.data.local_layout import (
    ALIASES_0B,
    LocalLayout,
    LocalLayoutError,
    normalize_name,
)

INDEX = load_class_index()

# Local folder names that differ from the normalised Keras class name.
LOCAL_NAMES = {
    134: "crane_bird",
    264: "cardigan_welsh_corgi",
    474: "cardigan",
    517: "crane",
    638: "maillot",
    639: "tank_suit",
}


def make_root(tmp_path):
    for index_0b in range(1000):
        name = LOCAL_NAMES.get(index_0b, normalize_name(INDEX.get(index_0b).name))
        (tmp_path / name).mkdir()
    return tmp_path


def test_normalize_name_collapses_punctuation_and_case():
    assert normalize_name("jack-o'-lantern") == "jack_o_lantern"
    assert normalize_name("  Black-and-Tan Coonhound ") == "black_and_tan_coonhound"
    assert normalize_name("Afghan_hound") == "afghan_hound"


def test_aliases_match_the_class_index():
    assert INDEX.get(ALIASES_0B["crane_bird"]).synset == "n02012849"
    assert INDEX.get(ALIASES_0B["crane"]).synset == "n03126707"
    assert INDEX.get(ALIASES_0B["cardigan_welsh_corgi"]).synset == "n02113186"
    assert INDEX.get(ALIASES_0B["cardigan"]).synset == "n02963159"
    assert INDEX.get(ALIASES_0B["jack_o_lantern"]).synset == "n03590841"
    assert INDEX.get(ALIASES_0B["tank_suit"]).synset == "n03710721"
    assert INDEX.get(ALIASES_0B["maillot"]).synset == "n03710637"


def test_complete_flat_root_maps_every_class_once(tmp_path):
    layout = LocalLayout(make_root(tmp_path), INDEX)

    assert len(layout.counts()) == 1000
    assert layout.folder(134).name == "crane_bird"
    assert layout.folder(517).name == "crane"
    assert layout.folder(264).name == "cardigan_welsh_corgi"
    assert layout.folder(639).name == "tank_suit"


def test_images_are_sorted_and_filtered(tmp_path):
    root = make_root(tmp_path)
    folder = root / normalize_name(INDEX.get(0).name)
    for name in ("002.JPEG", "000.jpg", "notes.txt", "001.png"):
        (folder / name).write_bytes(b"x")

    assert [path.name for path in layout_images(root, INDEX)] == [
        "000.jpg",
        "001.png",
        "002.JPEG",
    ]


def layout_images(root, index):
    return LocalLayout(root, index).images(0)


def test_missing_class_folder_fails_closed(tmp_path):
    root = make_root(tmp_path)
    (root / normalize_name(INDEX.get(0).name)).rmdir()

    with pytest.raises(LocalLayoutError, match="missing 1 class folders"):
        LocalLayout(root, INDEX)


def test_unmapped_folder_fails_closed(tmp_path):
    root = make_root(tmp_path)
    (root / "totally_unknown_species").mkdir()

    with pytest.raises(LocalLayoutError, match="maps to 0 classes"):
        LocalLayout(root, INDEX)


def test_duplicate_folder_fails_closed(tmp_path):
    root = make_root(tmp_path)
    (root / "tench!").mkdir()

    with pytest.raises(LocalLayoutError, match="both .* map to index 0"):
        LocalLayout(root, INDEX)
