import lib.net.layer as lyr
import lib.utils.array_utils as au
import lib.utils.dataset_utils as du
import numpy as np


class NeuralNet(object):

    def __init__(self, input_size, hidden_sizes, output_size):
        input_layer = lyr.Layer.build_input_layer(size=input_size)
        hidden_layers = lyr.Layer.build_hidden_layers(sizes=hidden_sizes, prev_layer_size=input_size)
        output_layer = lyr.Layer.build_output_layer(size=output_size, prev_layer_size=hidden_sizes[-1])

        self.layers = au.flatten([input_layer, hidden_layers, output_layer])

    def set_weights(self, weights):
        for i in range(1, len(self.layers)):
            self.layers[i].set_neurons_weights(weights[i-1])

    def get_weights(self):
        weights = []
        for i in range(1, len(self.layers)):
            weights.append([])
            for j in range(len(self.layers[i].neurons)):
                weights[i-1].append(self.layers[i].neurons[j].weights)
        return weights

    def feedforward(self, x_vec):
        layers_outputs = []
        for layer in self.layers:
            layers_outputs.append(layer.outputs(x_vec=x_vec))
            x_vec = layers_outputs[-1]

        return layers_outputs

    def predict(self, x_vec):
        net_outputs = self.feedforward(x_vec=x_vec)
        return net_outputs[-1]

    def backpropagation(self, layers_outputs, targets):
        output_layer = self.layers[-1]
        layers_errors_deltas = [output_layer.errors_deltas(outputs=layers_outputs[-1], targets=targets)]

        for i in range(len(self.layers)-2, 0, -1):
            hidden_layer = self.layers[i]
            errors_deltas = hidden_layer.errors_deltas(outputs=layers_outputs[i], next_layer=self.layers[i+1],
                                                       next_layer_deltas=layers_errors_deltas[0])
            layers_errors_deltas.insert(0, errors_deltas)

        return layers_errors_deltas

    def update_weights(self, layers_inputs, layers_error_deltas, lr, mmt):
        for i in range(1, len(self.layers)):
            self.layers[i].update_neurons_weights(inputs=layers_inputs[i-1], errors_deltas=layers_error_deltas[i-1],
                                                  lr=lr, mmt=mmt)

    def train(self, inputs, targets, epochs, lr=0.01, mmt=0.5, dlr=0.1, dlr_rate=0.1, logs_grad=10, val_split=0.2):
        [train_errors, test_errors] = [[], []]
        dec_step = dlr_rate*epochs
        inputs, targets = du.shuffle(inputs, targets)
        [[train_inputs, train_targets], [test_inputs, test_targets]] = du.split(inputs, targets, val_split)

        for i in range(epochs):
            x, t = du.shuffle(train_inputs, train_targets)
            errors = []

            if not (i % dec_step):
                lr *= (1.0-dlr)
                print('LR: ' + str(lr))

            for xi, ti in zip(x, t):
                layers_outputs = self.feedforward(x_vec=xi)
                layers_error_deltas = self.backpropagation(layers_outputs=layers_outputs, targets=ti)
                self.update_weights(layers_inputs=layers_outputs, layers_error_deltas=layers_error_deltas, lr=lr, mmt=mmt)

                outputs = np.array(layers_outputs[-1])
                error = 0.5*((outputs - np.array(ti)) ** 2).sum()
                errors.append(error)

            avg_train_error = sum(errors) / len(t)

            avg_test_error = 0
            if len(test_inputs):
                errors = []
                for xi, ti in zip(test_inputs, test_targets):
                    outputs = self.feedforward(x_vec=xi)[-1]
                    error = 0.5*((outputs - np.array(ti)) ** 2).sum()
                    errors.append(error)

                avg_test_error = sum(errors) / len(test_inputs)

            if not (i % logs_grad):
                print('Epoch: ' + str(i) + ', train error: ' + str(avg_train_error) +
                      ', test error: ' + str(avg_test_error))
                train_errors.append(avg_train_error)
                test_errors.append(avg_test_error)

        return [train_errors, test_errors]


    def interactiveTrain(self, inputs, targets, epochs, logger, lr=0.01, mmt=0.5, dlr=0.1, dlr_rate=0.1, val_split=0.2):
        [train_errors, test_errors] = [[], []]
        dec_step = dlr_rate*epochs
        inputs, targets = du.shuffle(inputs, targets)
        [[train_inputs, train_targets], [test_inputs, test_targets]] = du.split(inputs, targets, val_split)

        if (epochs > 200):
            logs_grad = epochs // 200;
        else:
            logs_grad = 1;

        for i in range(epochs):
            x, t = du.shuffle(train_inputs, train_targets)
            errors = []

            if not (i % dec_step):
                lr *= (1.0-dlr)
                logger("lr_update", { "lr": lr })

            for xi, ti in zip(x, t):
                layers_outputs = self.feedforward(x_vec=xi)
                layers_error_deltas = self.backpropagation(layers_outputs=layers_outputs, targets=ti)
                self.update_weights(layers_inputs=layers_outputs, layers_error_deltas=layers_error_deltas, lr=lr, mmt=mmt)

                outputs = np.array(layers_outputs[-1])
                error = 0.5*((outputs - np.array(ti)) ** 2).sum()
                errors.append(error)

            avg_train_error = sum(errors) / len(t)

            avg_test_error = 0
            if len(test_inputs):
                errors = []
                for xi, ti in zip(test_inputs, test_targets):
                    outputs = self.feedforward(x_vec=xi)[-1]
                    error = 0.5*((outputs - np.array(ti)) ** 2).sum()
                    errors.append(error)

                avg_test_error = sum(errors) / len(test_inputs)
            if not (i % logs_grad):
                logger("training_update", { "epoch": i, "train_error": avg_train_error, "test_error": avg_test_error })

            train_errors.append(avg_train_error)
            test_errors.append(avg_test_error)

        return [train_errors, test_errors]
