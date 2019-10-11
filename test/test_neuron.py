import unittest
import lib.net.neuron as nrn
import numpy as np


class NeuronTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_neuron_init(self):
        neuron = nrn.Neuron(input_size=3)
        max_weight = max(neuron.weights)
        min_weight = min(neuron.weights)
        self.assertEqual(len(neuron.weights), 4)
        self.assertLessEqual(max_weight, 1)
        self.assertGreater(min_weight, 0)

        neuron = nrn.Neuron(input_size=3).set_weights(weights=[1, 2, 3])
        self.assertEqual(neuron.weights.tolist(), [1, 2, 3])

    def test_sum(self):
        neuron = nrn.Neuron(input_size=3).set_weights(weights=[0, 0, 0, 0])
        self.assertEqual(neuron.sum(x_vec=[1, 1, 1]), 0)

        neuron = nrn.Neuron(input_size=3).set_weights(weights=[0.5, 1, 1, 1])  # bias x0 = 1.0
        self.assertEqual(neuron.sum(x_vec=[2, 3, 4]), 9.5)

        neuron = nrn.Neuron(input_size=3).set_weights(weights=[-0.5, 1, 1, 1])
        self.assertEqual(neuron.output(x_vec=[0.5, 0, 0]), 0.5)
        self.assertLessEqual(neuron.output(x_vec=[0, 10, 10]), 1)
        self.assertGreaterEqual(neuron.output(x_vec=[0, -10, -10]), 0)

    def test_calc_weights_delta(self):
        neuron = nrn.Neuron(input_size=2).set_weights(weights=[-0.5, 1, 1])
        self.assertEqual(neuron.calc_weights_deltas(x_vec=[2, 3], error_delta=-2).tolist(), [2, 4, 6])  # x0=1

    def test_update_weights(self):
        neuron = nrn.Neuron(input_size=3).set_weights(weights=[-0.5, 1, 1, 1])
        neuron.update_weights(weights_deltas=np.array([1, 1, 1, 1]), lr=1.0, mmt=0)
        self.assertEqual(neuron.weights.tolist(), [0.5, 2, 2, 2])

        neuron.update_weights(weights_deltas=np.array([1, 1, 1, 1]), lr=0.1, mmt=0)
        self.assertEqual(neuron.weights.tolist(), [0.6, 2.1, 2.1, 2.1])

        neuron = nrn.Neuron(input_size=3).set_weights(weights=[-0.5, 1, 1, 1])
        neuron.weights_deltas = np.array([-1, -4, -2, 0])
        neuron.update_weights(weights_deltas=np.array([1, 1, 1, 1]), lr=1.0, mmt=0.5)
        self.assertEqual(neuron.weights.tolist(), [0, 0, 1, 2])


class InputNeuronTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_output(self):
        neuron = nrn.InputNeuron(i=1)
        self.assertEqual(neuron.output(x_vec=[9, 8, 7]), 8)


class HiddenNeuronTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_calc_error_delta_hidden(self):
        neuron = nrn.HiddenNeuron(input_size=2).set_weights(weights=[0, -1, 1])
        output = neuron.output(x_vec=[0.5, 0.5])
        self.assertEqual(
            neuron.calc_error_delta(y=output, output_deltas_sum=1.5),
            0.375)


class OutputNeuronTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_calc_error_delta_output(self):
        neuron = nrn.OutputNeuron(input_size=2).set_weights(weights=[-0.5, -1, 1])
        output = neuron.output(x_vec=[0.5, 1])
        self.assertEqual(neuron.calc_error_delta(y=output, t=1), -0.125)
        output = neuron.output(x_vec=[0.5, 1])
        self.assertEqual(neuron.calc_error_delta(y=output, t=-0.5), 0.25)

if __name__ == '__main__':
    unittest.main()