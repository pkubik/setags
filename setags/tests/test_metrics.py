import numpy as np
from setags.metrics import confusion_matrix_from_sets, confusion_matrix_from_iterables, f1, ConfusionMatrix


TEST_TARGET = {1, 2, 3, 4, 5}
TEST_PREDICTION = {3, 4, 5, 6}


def test_single_example():
    cm = confusion_matrix_from_sets(TEST_TARGET, TEST_PREDICTION)

    assert cm.true_positive == 3
    assert cm.false_positive == 1
    assert cm.false_negative == 2

    assert cm.precision() == 0.75
    assert cm.recall() == 0.6


def test_iterable_of_examples():
    targets = (TEST_TARGET for _ in range(10))
    predictions = (TEST_PREDICTION for _ in range(10))

    cm = confusion_matrix_from_iterables(targets, predictions)

    assert cm.true_positive == 30
    assert cm.false_positive == 10
    assert cm.false_negative == 20

    assert cm.precision() == 0.75
    assert cm.recall() == 0.6


def test_empty_iterable():
    cm = confusion_matrix_from_iterables([], [])
    # noinspection PyTypeChecker
    assert np.all(cm == ConfusionMatrix())


def test_empty_confusion_matrix():
    cm = ConfusionMatrix()
    assert np.isnan(cm.precision())
    assert np.isnan(cm.recall())


def test_f1():
    precision = 0.2
    recall = 0.6
    f1_score = f1(precision, recall)
    sym_f1_score = f1(recall, precision)

    assert f1_score == sym_f1_score == 0.3
