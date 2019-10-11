import lib.net.neural_net as nnet
import unittest


class NeuralNetsTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        net = nnet.NeuralNet(input_size=2, hidden_sizes=[3], output_size=1)
        self.assertEqual(len(net.layers), 3)

        net = nnet.NeuralNet(input_size=2, hidden_sizes=[3, 2], output_size=1)
        self.assertEqual(len(net.layers), 4)

    def test_feedforward(self):
        net = nnet.NeuralNet(input_size=2, hidden_sizes=[2], output_size=1)
        net.set_weights(weights=[
            [[0, -1, 1], [0, 1, -1]],
            [[-1, 1, 1]]
        ])

        expected_outputs = [[0.5, 0.5], [0.5, 0.5], [0.5]]
        layers_outputs = net.feedforward(x_vec=[0.5, 0.5])
        for i, outputs in enumerate(layers_outputs):
            self.assertSequenceEqual(outputs, expected_outputs[i])

        net = nnet.NeuralNet(input_size=2, hidden_sizes=[3], output_size=2)
        net.set_weights(weights=[
            [[-0.75, 1, 1], [-0.5, 0, 1], [0.25, 1, -1]],
            [[0, -1, 1, 0], [0, 0, 0, 0]]
        ])

        expected_outputs = [[0.25, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5]]
        layers_outputs = net.feedforward(x_vec=[0.25, 0.5])
        for i, outputs in enumerate(layers_outputs):
            self.assertSequenceEqual(outputs, expected_outputs[i])

        net = nnet.NeuralNet(input_size=2, hidden_sizes=[1, 2], output_size=1)
        net.set_weights(weights=[
            [[0.25, 1, -1]],
            [[-0.25, 0.5], [0.5, -1]],
            [[0, 1, 1]]
        ])

        expected_outputs = [[0.5, 0.75], [0.5], [0.5, 0.5], [0.73]]
        layers_outputs = net.feedforward(x_vec=[0.5, 0.75])
        layers_outputs[-1][0] = round(layers_outputs[-1][0], 2)
        for i, outputs in enumerate(layers_outputs):
            self.assertSequenceEqual(outputs, expected_outputs[i])

    def test_predict(self):
        net = nnet.NeuralNet(input_size=2, hidden_sizes=[2], output_size=1)
        net.set_weights(weights=[
            [[0, -1, 1], [0, 1, -1]],
            [[-1, 1, 1]]
        ])

        self.assertSequenceEqual(net.predict(x_vec=[0.5, 0.5]), [0.5])

    def test_backpropagation(self):
        net = nnet.NeuralNet(input_size=2, hidden_sizes=[3], output_size=2)
        net.set_weights(weights=[
            [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
            [[0, 1, 1, 1], [0, 1, 1, 1]]
        ])

        layers_errors_deltas = net.backpropagation(layers_outputs=[[0, 0], [0.5, -0.5, 0.5], [0.5, 0.5]], targets=[0, 0.5])
        expected_errors_deltas = [[0.03125, -0.09375, 0.03125], [0.125, 0]]
        for i in range(len(layers_errors_deltas)):
            self.assertSequenceEqual(layers_errors_deltas[i], expected_errors_deltas[i])

    def test_update_weights(self):
        net = nnet.NeuralNet(input_size=2, hidden_sizes=[3], output_size=2)
        net.set_weights(weights=[
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 1]]
        ])

        layers_outputs = [[1, 2], [3, 4, 0], [5, 6]]
        layers_errors_deltas = [[1, -1, 1], [1, 0.5]]
        net.update_weights(layers_inputs=layers_outputs, layers_error_deltas=layers_errors_deltas, lr=1.0, mmt=0)

        expected_weights = [[[-1, -1, -2], [1, 1, 2], [-1, -1, -2]], [[-1, -3, -4, 0], [-0.5, -1.5, -2, 1]]]
        layers_weights = net.get_weights()
        for i in range(len(layers_weights)):
            for j in range(len(layers_weights[i])):
                self.assertSequenceEqual(layers_weights[i][j].tolist(), expected_weights[i][j])
