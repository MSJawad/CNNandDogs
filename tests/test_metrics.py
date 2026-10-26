import numpy as np
import pytest

from catdog.metrics import confusion_matrix, report


def test_confusion_matrix_rows_are_true_columns_are_predicted():
    cm = confusion_matrix([0, 0, 0, 1, 1], [0, 1, 0, 1, 1], num_classes=2)
    np.testing.assert_array_equal(cm, [[2, 1], [0, 2]])


def test_confusion_matrix_shape_mismatch():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0], num_classes=2)


def test_report_values():
    # 8 cats (6 right), 12 dogs (11 right)
    cm = np.array([[6, 2], [1, 11]])
    r = report(cm, ["cat", "dog"])
    assert r["n"] == 20
    assert r["accuracy"] == pytest.approx(17 / 20)
    assert r["per_class"]["cat"]["recall"] == pytest.approx(6 / 8)
    assert r["per_class"]["cat"]["precision"] == pytest.approx(6 / 7)
    assert r["per_class"]["dog"]["recall"] == pytest.approx(11 / 12)
    assert r["per_class"]["dog"]["support"] == 12
    assert r["balanced_accuracy"] == pytest.approx((6 / 8 + 11 / 12) / 2)
    p, rc = 6 / 7, 6 / 8
    assert r["per_class"]["cat"]["f1"] == pytest.approx(2 * p * rc / (p + rc))


def test_report_handles_a_class_that_is_never_predicted():
    r = report(np.array([[0, 3], [0, 5]]), ["cat", "dog"])
    assert r["per_class"]["cat"]["precision"] == 0.0
    assert r["per_class"]["cat"]["f1"] == 0.0
    assert r["accuracy"] == pytest.approx(5 / 8)
