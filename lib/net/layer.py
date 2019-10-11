import lib.net.neuron as nrn
import numpy as np
from typing import List


class Layer(object):

    def __init__(self, size: int) -> None:
        self.size = size
        self.neurons = [] # type: List[nrn.Neuron]

    def outputs(self, x_vec: np.ndarray):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output(x_vec))
        return outputs

    def update_neurons_weights(self, inputs: np.ndarray, errors_deltas: np.ndarray, lr: float, mmt: float):
        for i in range(len(self.neurons)):
            weights_deltas = self.neurons[i].calc_weights_deltas(x_vec=inputs, error_delta=errors_deltas[i])
            self.neurons[i].update_weights(weights_deltas=weights_deltas, lr=lr, mmt=mmt)

    def set_neurons_weights(self, weights: List[List[float]]):
        for i in range(len(self.neurons)):
            self.neurons[i].set_weights(weights=weights[i])
        return self

    @staticmethod
    def build_input_layer(size: int) -> 'Layer':
        input_layer = Layer(size=size)
        for i in range(size):
            input_layer.neurons.append(nrn.InputNeuron(i))
        return input_layer

    @staticmethod
    def build_hidden_layer(size: int, prev_layer_size: int) -> 'HiddenLayer':
        hidden_layer = HiddenLayer(size=size)
        for i in range(size):
            hidden_layer.neurons.append(nrn.HiddenNeuron(input_size=prev_layer_size))
        return hidden_layer

    @staticmethod
    def build_hidden_layers(sizes: List[int], prev_layer_size: int) -> List['HiddenLayer']:
        prev_layer_sizes = [prev_layer_size] + sizes
        hidden_layers = []
        for i, layer_size in enumerate(sizes):
            hidden_layers.append(Layer.build_hidden_layer(size=layer_size, prev_layer_size=prev_layer_sizes[i]))
        return hidden_layers

    @staticmethod
    def build_output_layer(size: int, prev_layer_size: int) -> 'OutputLayer':
        output_layer = OutputLayer(size=size)
        for i in range(size):
            output_layer.neurons.append(nrn.OutputNeuron(input_size=prev_layer_size))
        return output_layer


class OutputLayer(Layer):

    def __init__(self, size: int) -> None:
        Layer.__init__(self, size=size)

    def errors_deltas(self, outputs: List[float], targets: List[int]) -> List[float]:
        errors_deltas = []
        for i in range(len(self.neurons)):
            errors_deltas.append(self.neurons[i].calc_error_delta(y=outputs[i], t=targets[i]))

        return errors_deltas


class HiddenLayer(Layer):

    def __init__(self, size: int) -> None:
        Layer.__init__(self, size=size)

    def errors_deltas(self,
                      outputs: List[float],
                      next_layer: 'Layer',
                      next_layer_deltas: List[float]) -> List[np.float64]:
        errors_deltas_sums = np.zeros(len(self.neurons))
        for i in range(len(next_layer.neurons)):
            errors_deltas_sums += next_layer_deltas[i]*next_layer.neurons[i].weights[1:]

        errors_deltas = []
        for i in range(len(self.neurons)):
            errors_deltas.append(self.neurons[i].calc_error_delta(y=outputs[i], output_deltas_sum=errors_deltas_sums[i]))

        return errors_deltas
