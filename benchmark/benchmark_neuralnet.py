import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('./'))
import lib.net.neural_net as nnet


def prepare_data():
    #print('Downloading iris data...')
    #df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    print('Using iris data...')
    df = pd.read_csv('data/iris.data', header=None)
    # print(df.tail())

    targets = df.iloc[:, 4].values
    mapping = {'Iris-virginica': [1, 0, 0], 'Iris-versicolor': [0, 1, 0], 'Iris-setosa': [0, 0, 1]}
    targets = list(map(lambda x: mapping[x], targets))
    inputs = df.iloc[:, :4].values
    #print(inputs, targets)

    return [inputs, targets]


def benchmark_lr():
    [inputs, targets] = prepare_data()

    print('Training for different learning rates...')
    net = nnet.NeuralNet(4, [6], 3)
    lrs = [0.01, 0.1, 1.0]
    colors = ['b', 'g', 'r']
    labels = ['lr = 0.01', 'lr = 0.1', 'lr = 1.0']
    logs_grad = 5

    for i in range(len(lrs)):
        [net, [train_errors, test_errors]] = fit(net,
                                                 inputs=inputs,
                                                 targets=targets,
                                                 epochs=250,
                                                 lr=lrs[i],
                                                 mmt=0.5,
                                                 dlr=0.0,
                                                 dlr_rate=1.0,
                                                 logs_grad=logs_grad,
                                                 val_split=0.0)

        x = [x * logs_grad for x in range(1, len(train_errors) + 1)]
        plt.plot(x, train_errors, marker='o', color=colors[i], label=labels[i])

    #plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.legend()
    plt.show()


def benchmark_mmt():
    [inputs, targets] = prepare_data()

    print('Training for different momentum values...')
    net = nnet.NeuralNet(4, [6], 3)
    mmts = [0.1, 0.5, 0.9]
    colors = ['b', 'g', 'r']
    labels = ['mmt = 0.1', 'mmt = 0.5', 'mmt = 0.9']
    logs_grad = 5

    for i in range(len(mmts)):
        [net, [train_errors, test_errors]] = fit(net,
                                                 inputs=inputs,
                                                 targets=targets,
                                                 epochs=250,
                                                 lr=0.01,
                                                 mmt=mmts[i],
                                                 dlr=0.0,
                                                 dlr_rate=1.0,
                                                 logs_grad=logs_grad,
                                                 val_split=0.0)

        x = [x * logs_grad for x in range(1, len(train_errors) + 1)]
        plt.plot(x, train_errors, marker='o', color=colors[i], label=labels[i])

    # plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.legend()
    plt.show()


def benchmark_dlr():
    [inputs, targets] = prepare_data()

    print('Training for different learning rate decreasing values...')
    net = nnet.NeuralNet(4, [6], 3)
    dlrs = [0.0, 0.1, 0.9]
    colors = ['b', 'g', 'r']
    labels = ['dlr = 0.0', 'dlr = 0.1', 'dlr = 0.9']
    logs_grad = 5

    for i in range(len(dlrs)):
        [net, [train_errors, test_errors]] = fit(net,
                                                 inputs=inputs,
                                                 targets=targets,
                                                 epochs=250,
                                                 lr=0.5,
                                                 mmt=0.9,
                                                 dlr=dlrs[i],
                                                 dlr_rate=0.1,
                                                 logs_grad=logs_grad,
                                                 val_split=0.0)

        x = [x * logs_grad for x in range(1, len(train_errors) + 1)]
        plt.plot(x, train_errors, marker='o', color=colors[i], label=labels[i])

    # plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.legend()
    plt.show()


def fit(net, inputs, targets, epochs, lr, mmt, dlr, dlr_rate, logs_grad, val_split):
    errors = net.train(inputs, targets, epochs, lr, mmt, dlr, dlr_rate, logs_grad, val_split)
    return [net, errors]


def benchmark():
    [inputs, targets] = prepare_data()
    val_split = 0.2

    print('Training...')
    net = nnet.NeuralNet(4, [6], 3)
    logs_grad = 5
    [net, [train_errors, test_errors]] = fit(net,
                                             inputs=inputs,
                                             targets=targets,
                                             epochs=200,
                                             lr=0.1,
                                             mmt=0.5,
                                             dlr=0.5,
                                             dlr_rate=0.1,
                                             logs_grad=logs_grad,
                                             val_split=val_split)

    x = [x * logs_grad for x in range(1, len(train_errors) + 1)]
    plt.plot(x, train_errors, marker='o', color='b', label='training error')
    if round(val_split, 5):
        x = [x * logs_grad for x in range(1, len(test_errors) + 1)]
        plt.plot(x, test_errors, marker='o', color='r', label='testing error')
    #plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    benchmark()
    #benchmark_lr()
    #benchmark_mmt()
    #benchmark_dlr()
