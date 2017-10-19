from typing import Iterable

import numpy as np


POSITIVE = 0
NEGATIVE = 1


class ConfusionMatrix(np.ndarray):
    def __new__(cls, **kwargs):
        obj = np.zeros([2, 2], dtype=np.int32).view(cls)
        return obj

    def __init__(self):
        pass

    @property
    def true_positive(self):
        return self[POSITIVE][POSITIVE]

    @true_positive.setter
    def true_positive(self, value):
        self[POSITIVE][POSITIVE] = value

    @property
    def false_positive(self):
        return self[POSITIVE][NEGATIVE]

    @false_positive.setter
    def false_positive(self, value):
        self[POSITIVE][NEGATIVE] = value

    @property
    def false_negative(self):
        return self[NEGATIVE][POSITIVE]

    @false_negative.setter
    def false_negative(self, value):
        self[NEGATIVE][POSITIVE] = value

    @property
    def true_negative(self):
        return self[NEGATIVE][NEGATIVE]

    @true_negative.setter
    def true_negative(self, value):
        self[NEGATIVE][NEGATIVE] = value

    def precision(self):
        return self.true_positive / (self.true_positive + self.false_positive)

    def recall(self):
        return self.true_positive / (self.true_positive + self.false_negative)


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall)


def confusion_matrix_from_sets(target: set, prediction: set) -> ConfusionMatrix:
    ret = ConfusionMatrix()
    ret.true_positive = len(target & prediction)
    ret.false_positive = len(prediction - target)
    ret.false_negative = len(target - prediction)
    ret.true_negative = 0
    return ret


def confusion_matrix_from_iterables(targets: Iterable[set], predictions: Iterable[set]) -> ConfusionMatrix:
    return sum((confusion_matrix_from_sets(target, prediction)
                for target, prediction in zip(targets, predictions)),
               ConfusionMatrix())
