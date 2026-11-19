from unlearning.classes import DOG_1B, CANID_1B, Group, load_class_index


def test_index_has_expected_classes_and_groups():
    index = load_class_index()

    assert index.get(0).imagenet_index_1b == 1
    assert index.get(999).imagenet_index_1b == 1000
    assert len(index.indices(Group.DOG)) == 118
    assert len(index.indices(Group.CANID)) == 7
    assert len(index.indices(Group.OTHER)) == 875
    assert index.indices(Group.DOG) == tuple(range(150, 268))
    assert index.indices(Group.CANID) == tuple(range(268, 275))
    assert len(index.indices(Group.CANID) + index.indices(Group.OTHER)) == 882


def test_group_boundaries():
    index = load_class_index()

    assert index.group(149) == Group.OTHER  # 1-based 150: sea lion
    assert index.group(150) == Group.DOG  # 1-based 151: Chihuahua
    assert index.group(267) == Group.DOG  # 1-based 268: Mexican hairless
    assert index.group(268) == Group.CANID  # 1-based 269: timber wolf
    assert index.group(274) == Group.CANID  # 1-based 275: African hunting dog
    assert index.group(275) == Group.OTHER  # 1-based 276: hyena


def test_synset_mapping_check():
    from unlearning.classes import check_synset_mapping

    index = load_class_index()
    synsets = {info.synset for info in index._by_index.values()}

    assert check_synset_mapping(index, synsets) == []
    assert check_synset_mapping(index, synsets - {"n01440764"}) == ["n01440764"]


def test_range_constants():
    assert len(DOG_1B) == 118
    assert len(CANID_1B) == 7
