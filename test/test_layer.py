import unittest
import lib.net.neuron as nrn
import lib.net.layer as lyr
import numpy as np


class LayerTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        layer = lyr.Layer(size=3)
        self.assertEqual(layer.neurons, [])

    def test_build_input_layer(self):
        layer = lyr.Layer.build_input_layer(2)
        self.assertEqual(layer.size, 2)
        self.assertEqual(len(layer.neurons), 2)
        self.assertListEqual(layer.neurons, [nrn.InputNeuron(0), nrn.InputNeuron(1)])

    def test_build_hidden_layer(self):
        layer = lyr.Layer.build_hidden_layer(size=2, prev_layer_size=1)
        self.assertEqual(layer.size, 2)
        self.assertEqual(len(layer.neurons), 2)
        for neuron in layer.neurons:
            self.assertEqual(len(neuron.weights), 1+1)

    def test_build_hidden_layers(self):
        hidden_layers = lyr.Layer.build_hidden_layers(sizes=[3], prev_layer_size=2)
        self.assertEqual(len(hidden_layers), 1)
        self.assertEqual(len(hidden_layers[0].neurons), 3)

        hidden_layers = lyr.Layer.build_hidden_layers(sizes=[1, 3], prev_layer_size=4)
        self.assertEqual(len(hidden_layers), 2)
        self.assertEqual(len(hidden_layers[0].neurons), 1)
        for neuron in hidden_layers[0].neurons:
            self.assertEqual(len(neuron.weights), 4+1)
        self.assertEqual(len(hidden_layers[1].neurons), 3)
        for neuron in hidden_layers[1].neurons:
            self.assertEqual(len(neuron.weights), 1+1)

    def test_build_output_layer(self):
        layer = lyr.Layer.build_output_layer(size=3, prev_layer_size=2)
        self.assertEqual(layer.size, 3)
        self.assertEqual(len(layer.neurons), 3)
        for neuron in layer.neurons:
            self.assertEqual(len(neuron.weights), 2+1)

    def test_update_neurons_weights(self):
        layer = lyr.Layer.build_output_layer(size=3, prev_layer_size=2)
        layer.set_neurons_weights(weights=[[1, 2, 3], [1, 4, 5], [0, 1, 2]])

        layer.update_neurons_weights(inputs=[1, 2], errors_deltas=[1, 2, 3], lr=1, mmt=0)
        expected_weights = [[0, 1, 1], [-1, 2, 1], [-3, -2, -4]]
        for i in range(len(layer.neurons)):
            self.assertSequenceEqual(layer.neurons[i].weights.tolist(), expected_weights[i])

        layer = lyr.Layer.build_output_layer(size=3, prev_layer_size=2)
        layer.set_neurons_weights(weights=[[1, 2, 3], [1, 4, 5], [0, 1, 2]])
        layer.update_neurons_weights(inputs=[1, 2], errors_deltas=[1, 2, 3], lr=0.5, mmt=0)
        expected_weights = [[0.5, 1.5, 2], [0, 3, 3], [-1.5, -0.5, -1]]
        for i in range(len(layer.neurons)):
            self.assertSequenceEqual(layer.neurons[i].weights.tolist(), expected_weights[i])

    def test_output_layer_error_deltas(self):
        output_layer = lyr.Layer.build_output_layer(size=2, prev_layer_size=2)
        errors_deltas = np.round(output_layer.errors_deltas(outputs=[0.5, 0.1], targets=[1, 0]), 3).tolist()
        self.assertSequenceEqual(errors_deltas, [-0.125, 0.009])

    def test_hidden_layer_error_deltas(self):
        hidden_layer = lyr.Layer.build_hidden_layer(size=2, prev_layer_size=2)
        next_layer = lyr.Layer.build_hidden_layer(size=2, prev_layer_size=2)
        next_layer.set_neurons_weights([[0, 1, 1], [0, 0, -1]])
        errors_deltas = hidden_layer.errors_deltas(outputs=[0.5, -0.5], next_layer=next_layer,
                                                   next_layer_deltas=[-0.5, 0.5])
        self.assertSequenceEqual(errors_deltas, [-0.125, 0.75])

        next_layer = lyr.Layer.build_hidden_layer(size=3, prev_layer_size=2)
        next_layer.set_neurons_weights([[0, 1, 1], [0, 0, -1], [0, 1, 1]])
        errors_deltas = hidden_layer.errors_deltas(outputs=[0.5, -0.5], next_layer=next_layer,
                                                   next_layer_deltas=[-0.5, 0.5, 1])
        self.assertSequenceEqual(errors_deltas, [0.125, 0])

