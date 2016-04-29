import numpy as np
from neuron import SimpleNeuron
from helpers.diagnostic_helpers import time_measure

class NeuralNetwork(object):
    def __init__(self, layer_size=[2, 2, 1], activation_function=None, activation_fun_prime=None):
        """
        Default activation function: np.tanh(x)
        Default learning function: self.eta * sigma * x * ((1 - x) * (1 + x))
        Other examples of functions:
            sigmoid: 1 / (1 + np.exp(-x))
            sigmoid prime: np.exp(-x) / ((1 + np.exp(-x)) ** 2)
            
        :param layer_size (optional) - list that defines number of neurons of each layer; default value: [2, 2, 1]
        :param activation_function (optional) - lambda expression; default value: lambda x: np.tanh(x) 
        :param activation_fun_prime (optional) - lambda expression; default value: lambda x: (1 - x) * (1 + x)
        """
        self._layer_size = layer_size
        self._layers = list()
        self.verbose = True
        self.eta = 2 * 0.05  # teaching speed
        """
        Other examples of activation function:
            Sigmoid: 1 / (1 + np.exp(-x))
            np.tanh(x)
        """
        self.activation_function = activation_function
        if self.activation_function is None:
            self.activation_function = lambda x: np.tanh(x)
        """
        Other examples of activation function prime:
            Sigmoid Prime: np.exp(-x) / ((1 + np.exp(-x)) ** 2)
            (1 - x) * (1 + x)
        """
        self.activation_fun_prime = activation_fun_prime
        if self.activation_fun_prime is None:
            self.activation_fun_prime = lambda x: (1 - x) * (1 + x)  # derivative of activation_function
        self.learning_function = lambda x, y, sigma: self.eta * sigma * x * self.activation_fun_prime(x)

        for i in range(len(self._layer_size)):
            layer_i = list()
            for j in range(self._layer_size[i]):
                if i == 0:
                    neuron = SimpleNeuron(1, weights=[1],
                                          activation_function=self.activation_function)
                else:
                    neuron = SimpleNeuron(len(self._layers[i - 1]), activation_function=self.activation_function)
                layer_i.append(neuron)
            self._layers.append(layer_i)

    def __repr__(self):
        str_ = ""
        for i in range(len(self._layers)):
            str_ += "Layer {0}:\r\n".format(i)
            for j in range(len(self._layers[i])):
                str_ += "\t{0}. {1}\r\n".format(j, self._layers[i][j])
        return str_

    def _print(self, *args):
        if self.verbose:
            print(" ".join([str(x) for x in args]))

    def last_layer_idx(self):
        """
        Gets index of last layer
        :return index of last layer
        """
        return len(self._layers) - 1

    def get_neurons_count(self, layer_idx):
        """
        Gets neurons count on selected layer
        :param layer_idx - layer index
        :return neurons count 
        """
        return len(self._layers[layer_idx])

    def get_layers_count(self):
        """
        Gets layers count
        :return layers count 
        """
        return len(self._layers)

    def set_neurons_weights(self, layer_idx, neuron_idx, weights):
        """
        Sets weights of selected neuron
        :param layer_idx - layer index
        :param neuron_idx - neuron index
        :param weights - list of weights (ints/floats)
        """
        self._layers[layer_idx][neuron_idx].weights = weights

    def get_neuron(self, layer_idx, neuron_idx):
        """
        Gets Neuron
        :param layer_idx - layer index
        :param neuron_idx - neuron index
        :return Neuron object
        """
        return self._layers[layer_idx][neuron_idx]

    def get_neurons(self, layer_idx):
        """
        Gets Neurons from seleced layer
        :param layer_idx - layer index
        :return list with Neurons
        """
        return self._layers[layer_idx]

    def get_outputs(self, layer_idx):
        """
        Gets outputs of selected layer
        :param layer_idx - layer index
        :return outputs - neurons outputs as list of np.arrays
        """
        outputs = list()
        for i in range(0, self.get_neurons_count(layer_idx)):
            neuron = self.get_neuron(layer_idx, i)
            outputs.append(neuron.last_y)
        return outputs

    def get_weights(self, layer_idx):
        """
        Gets weights of selected layer
        :param layer_idx - layer index
        :return weights - neurons weights as list of np.arrays
        """
        weights = list()
        for i in range(0, self.get_neurons_count(layer_idx)):
            neuron = self.get_neuron(layer_idx, i)
            weights.extend(neuron.weights)
        return weights

    def _get_neuron_weights(self, layer_idx, neuron_idx):
        """
        Gets neurons weights
        :param layer_idx - layer index
        :param neuron_idx - neuron index
        :return weights - neurons weights
        """
        neuron = self.get_neuron(layer_idx, neuron_idx)
        return neuron.weights

    def _run_first_layer(self, x):
        """
        Runs first layer, that only contains weights for inputs. Returned array has the same size as input array x.
        :param x - data as np.array
        :return y- output of first layer as np.array
        """
        weights = list()
        for i in range(len(self._layers[0])):
            weights.extend(self._get_neuron_weights(0, i))
        y = np.multiply(x, np.array((weights)))
        return y

    def _run_layer(self, x, layer_idx):
        """
        Runs selected layer
        :param x - input data as np.array
        :param layer_idx - layer index
        :return y - output of selected layer as np.array
        """
        y = list()
        for neuron in self._layers[layer_idx]:
            y.append(neuron.compute(x))
        return np.array(y)

    def apply_changes(self, from_layer, to_layer):
        """
        Applies changes to neurons after learning process
        :param from_layer
        :param to_layer
        """
        for layer_idx in range(from_layer, to_layer):  # do not apply change to first layer
            for neuron_idx in range(self.get_neurons_count(layer_idx)):
                self._layers[layer_idx][neuron_idx].apply_changes()

    @time_measure
    def run(self, x):
        """
        Run network
        :param x - input data as np.array
        :return y - output value as int/float
        """
        if len(x) != len(self._layers[0]):
            raise Exception("First layer has to have {0} neurons!".format(len(x)))

        self._print('Running network...')
        y_1 = self._run_first_layer(x)
        self._print('y_0:', y_1)
        y_i = y_1
        for i in range(len(self._layers) - 1):
            i += 1
            y_i = self._run_layer(y_i, i)
            self._print('y_{0}:'.format(i), y_i)
        return np.array(y_i)

    @time_measure
    def learn(self, x, z):
        """
        Learning using back propagation.
        :param x - input data as np.array
        :param z - desired output value as int/float
        """
        self._print('Learning...')
        y = self.run(x)
        sigma = z - y

        # last layer:
        for neuron_idx in range(self.get_neurons_count(self.last_layer_idx())):
            self._layers[self.last_layer_idx()][neuron_idx].sigma = sigma
            self._layers[self.last_layer_idx()][neuron_idx].delta_w = self.learning_function(self._layers[self.last_layer_idx()][neuron_idx].last_x, y, sigma)

        for layer_idx in range(self.get_layers_count() - 2, -1, -1):
            for neuron_idx in range(len(self._layers[layer_idx])):
                selected_neuron = self._layers[layer_idx][neuron_idx]
                sigm_sum = 0
                # sigma for n-th layer is a sum of sigma's from n+1 layer multiplied by proper weights from n+1 layer
                for neuron_idx_2 in range(len(self._layers[layer_idx + 1])):
                    sigm_sum += self._layers[layer_idx + 1][neuron_idx_2].sigma * self._layers[layer_idx + 1][neuron_idx_2].weights[neuron_idx]
                selected_neuron.sigma = sigm_sum
                selected_neuron.delta_w = self.learning_function(
                                    selected_neuron.last_x, selected_neuron.last_y, sigma)
                selected_neuron.delta_bias = self.learning_function(
                                    1,
                                    selected_neuron.bias, sigma)

        self.apply_changes(1, self.get_layers_count())  # do not apply change to first layer

    @time_measure
    def hebb_learn(self, x, eta=0.5):
        """
        Learning using Hebbian learning rule.
        :param x - input data as np.array
        """
        self._print('Learning with Hebbian learning rule...')
        self.run(x)

        for layer_idx in range(0, self.get_layers_count()):
            for neuron_idx in range(self.get_neurons_count(layer_idx)):
                delta_w = list()
                neuron = self._layers[layer_idx][neuron_idx]
                for i in range(len(neuron.weights)):
                    delta_w.append(neuron.weights[i] * eta * neuron.last_y)
                neuron.delta_w = delta_w

        self.apply_changes(1, self.get_layers_count())  # do not apply change to first layer

    @time_measure
    def kohonen_learn(self, x):
        """
        Learning using Kohonen method.
        :param x - input data as np.array
        """
        x_norm = np.linalg.norm(x)
        raise NotImplementedError()


def example():
    # init network - 2 inputs, 2 Neurons in hidden layer and 1 neuron for output
    net = NeuralNetwork(layer_size=[2, 2, 1])

    # setting weights:
    net.set_neurons_weights(1, 0, np.array(([0, 1]), dtype=float))
    net.set_neurons_weights(1, 1, np.array(([1, 0]), dtype=float))
    net.set_neurons_weights(2, 0, np.array(([1, 1]), dtype=float))

    x = np.array(([-1, 1]), dtype=float)
    print 'input data:', x
    net.run(x)
    net.learn(x, np.array([0.5]))
    print net


def example_hebb():
    # init network - 2 inputs, 2 Neurons in hidden layer and 1 neuron for output
    net = NeuralNetwork(layer_size=[2, 2, 1])

    # setting weights:
    net.set_neurons_weights(1, 0, np.array(([0, 1]), dtype=float))
    net.set_neurons_weights(1, 1, np.array(([1, 0]), dtype=float))
    net.set_neurons_weights(2, 0, np.array(([1, 1]), dtype=float))

    x = np.array(([-1, 1]), dtype=float)
    print 'input data:', x
    net.run(x)
    print net
    net.hebb_learn(x)
    print net

