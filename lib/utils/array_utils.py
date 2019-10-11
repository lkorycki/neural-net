import collections


def flatten(x):
    if isinstance(x, collections.Iterable) and not isinstance(x, str):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
