import math


def sigmoid(x):
    try:
        res = 1 / (1 + math.exp(-x))
    except OverflowError:
        res = 1 if x > 0 else 0

    return res