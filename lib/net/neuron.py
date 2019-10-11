import lib.utils.math_utils as mu
import lib.utils.dataset_utils as du
import numpy as np
from typing import List, Tuple


class Neuron(object):

    def __init__(self, input_size: int) -> None:
        self.weights = np.random.rand(1 + input_size)
        self.weights_deltas = np.zeros(1 + input_size)

    def sum(self, x_vec: np.ndarray):
        return np.dot(x_vec, self.weights[1:]) + 1*self.weights[0]

    def output(self, x_vec: np.ndarray):
        return mu.sigmoid(self.sum(x_vec))

    def update_weights(self, weights_deltas: np.ndarray, lr: float=0.01, mmt: float=0.5):
        self.weights += lr*weights_deltas
        self.weights += mmt*self.weights_deltas
        self.weights_deltas = weights_deltas
        return self

    def set_weights_deltas(self, weights_deltas: List[float]):
        self.weights_deltas = np.array(weights_deltas)
        return self

    def set_weights(self, weights: List[float]):
        self.weights = np.array(weights, dtype=np.float64)
        return self

    @staticmethod
    def calc_weights_deltas(x_vec: np.ndarray, error_delta: np.float64):
        return (-1) * error_delta * np.append([1], x_vec)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class OutputNeuron(Neuron):

    def __init__(self, input_size: int) -> None:
        Neuron.__init__(self, input_size)

    def calc_error_delta(self, y: float, t: int) -> float:
        return -(t-y)*y*(1-y)

    def train(self,
              inputs: np.ndarray,
              targets: np.ndarray,
              epochs: int,
              lr: float=0.01,
              mmt: float=0.5,
              dlr: float=0.1,
              dlr_rate: float=0.1,
              logs_grad: int=10,
              val_split: float=0.1) -> Tuple[List[np.float64], List[np.float64]]:
        [train_errors, test_errors] = [[], []]
        dec_step = dlr_rate * epochs
        inputs, targets = du.shuffle(inputs, targets)
        [[train_inputs, train_targets], [test_inputs, test_targets]] = du.split(inputs, targets, val_split)

        for i in range(epochs):
            x, t = du.shuffle(train_inputs, train_targets)
            errors = []

            if not (i % dec_step):
                lr *= (1.0-dlr)
                print('LR decreased: ' + str(lr))

            for xi, ti in zip(x, t):
                output = self.output(xi)
                error_delta = self.calc_error_delta(y=output, t=ti)
                weights_deltas = self.calc_weights_deltas(x_vec=xi, error_delta=error_delta)
                self.update_weights(weights_deltas=weights_deltas, lr=lr, mmt=mmt)

                error = ti - output
                errors.append(0.5 * (error ** 2))

            avg_train_error = sum(errors) / len(t)

            avg_test_error = 0
            if len(test_inputs):
                errors = []
                for xi, ti in zip(test_inputs, test_targets):
                    output = self.output(x_vec=xi)
                    error = ti - output
                    errors.append(0.5 * (error ** 2))

                avg_test_error = sum(errors) / len(test_inputs)

            if not (i % logs_grad):
                print('Epoch: ' + str(i) + ', train error: ' + str(avg_train_error) +
                      ', test error: ' + str(avg_test_error))
                train_errors.append(avg_train_error)
                test_errors.append(avg_test_error)

        return (train_errors, test_errors)


class HiddenNeuron(Neuron):

    def __init__(self, input_size: int) -> None:
        Neuron.__init__(self, input_size)

    def calc_error_delta(self, y: float, output_deltas_sum: float) -> float:
        return y*(1-y)*output_deltas_sum


class InputNeuron(Neuron):

    def __init__(self, i: int) -> None:
        self.i = i

    def output(self, x_vec: np.ndarray):
        return x_vec[self.i]

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
