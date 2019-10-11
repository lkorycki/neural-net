import sklearn.utils as su


def shuffle(x, y):
    return su.shuffle(x, y, random_state=0)


def split(x, y, val_split):
    p = int(val_split * len(x))
    train_set = [x[p:], y[p:]]
    test_set = [x[:p], y[:p]]

    return [train_set, test_set]
