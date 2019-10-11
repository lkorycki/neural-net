import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('./'))
import lib.net.neuron as neuron


def fit(inputs, targets, epochs, lr, mmt, dlr, dlr_rate, logs_grad, val_split):
    nrn = neuron.OutputNeuron(input_size=2)
    errors = nrn.train(inputs, targets, epochs, lr, mmt, dlr, dlr_rate, logs_grad, val_split)
    return [nrn, errors]


def benchmark():
    print('Downloading iris data...')
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    #print(df.tail())

    targets = df.iloc[0:100, 4].values
    targets = np.where(targets == 'Iris-versicolor', 0, 1)
    inputs = df.iloc[0:100, [0, 2]].values
    #print y_vec, x_vec

    print('Training...')
    val_split = 0.2
    [nrn, [train_errors, test_errors]] = fit(inputs, targets, epochs=50, lr=0.01, mmt=0.5,
                                             dlr=0.0, dlr_rate=1.0, logs_grad=1, val_split=val_split)
    plt.plot(range(1, len(train_errors) + 1), train_errors, marker='o', color='b', label='training error')
    if round(val_split, 5):
        plt.plot(range(1, len(test_errors) + 1), test_errors, marker='o', color='r', label='testing error')
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    benchmark()