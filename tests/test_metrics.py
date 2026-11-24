import numpy as np

from unlearning.classes import load_class_index
from unlearning.metrics import evaluate, probabilities

INDEX = load_class_index()


def test_probabilities_sum_to_one():
    probs = probabilities(np.array([[1.0, 2.0, 3.0]]))
    assert np.isclose(probs.sum(axis=-1), 1.0).all()
    assert (probs >= 0).all()


def test_evaluate_reports_dog_retain_and_canid_metrics():
    labels = np.array([150, 150, 268, 0, 0])
    logits = np.full((5, 1000), -10.0)
    logits[0, 150] = 5.0  # correct dog
    logits[1, 0] = 5.0  # dog mistaken for tench
    logits[2, 268] = 5.0  # correct canid
    logits[3, 0] = 5.0  # correct retain
    logits[4, 150] = 5.0  # retain mistaken for dog

    metrics = evaluate(logits, labels, INDEX, teacher_logits=logits)

    assert metrics["forget_top1"] == 0.5
    assert metrics["any_dog_top1"] == 0.5
    assert metrics["retain_micro_top1"] == 2 / 3  # retain includes the canid sample
    assert metrics["canid_top1"] == 1.0
    assert metrics["canid_top5"] == 1.0
    assert metrics["canid_mass"] > 0.9
    assert "tench" in metrics["top_dog_confusions"]
    assert metrics["plausibility_rate"] == 1.0
    assert 0.0 < metrics["entropy"] < 3.0


def test_empty_groups_do_not_divide_by_zero():
    labels = np.array([0, 1])
    logits = np.zeros((2, 1000))
    metrics = evaluate(logits, labels, INDEX)

    assert metrics["forget_top1"] == 0.0
    assert metrics["canid_mass"] == 0.0
    assert "top_dog_confusions" not in metrics
