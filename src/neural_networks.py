import numpy as np
from neuron import SimpleNeuron
from helpers.diagnostic_helpers import time_measure

class NeuralNetwork(object):
    def __init__(self, input_size=1, layer_size=[2, 2, 1]):
        """
        Default activation function: np.tanh(x)
        Default learning function: self.eta * sigma * x * ((1 - x) * (1 + x))
        Other examples of functions:
            sigmoid: 1 / (1 + np.exp(-x))
            sigmoid prime: np.exp(-x) / ((1 + np.exp(-x)) ** 2)
        """
        self.verbose = True
        self.layer_size = layer_size
        self.layers = list()
        self.eta = 2 * 0.05 # teaching speed
        """
        Other examples of activation function:
            Sigmoid: 1 / (1 + np.exp(-x))
            np.tanh(x)
        """
        self.activation_function = lambda x: np.tanh(x)
        """
        Other examples of activation function prime:
            Sigmoid Prime: np.exp(-x) / ((1 + np.exp(-x)) ** 2)
            (1 - x) * (1 + x)
        """
        self.activation_fun_prime = lambda x: (1 - x) * (1 + x)
        self.learning_function = lambda x, y, sigma: self.eta * sigma * x * self.activation_fun_prime(x)  # derivative of activation_function

        for i in range(len(layer_size)):
            layer_i = list()
            for j in range(layer_size[i]):
                if i == 0:
                    neuron = SimpleNeuron(input_size, input_size, activation_function=self.activation_function)
                else:
                    neuron = SimpleNeuron(len(self.layers[i - 1]), activation_function=self.activation_function)
                layer_i.append(neuron)
            self.layers.append(layer_i)

    def __repr__(self):
        str_ = ""
        for i in range(len(self.layers)):
            str_ += "Layer {0}:\r\n".format(i)
            for j in range(len(self.layers[i])):
                str_ += "\t{0}. {1}\r\n".format(j, self.layers[i][j])
        return str_

    def _print(self, *args):
        if self.verbose:
            print(" ".join([str(x) for x in args]))

    def last_layer_idx(self):
        """
        Gets index of last layer
        :return index of last layer
        """
        return len(self.layers) - 1

    def get_neurons_count(self, layer_idx):
        """
        Gets neurons count on selected layer
        :param layer_idx - layer index
        :return neurons count 
        """
        return len(self.layers[layer_idx])

    def get_layers_count(self):
        """
        Gets layers count
        :return layers count 
        """
        return len(self.layers)

    def set_neurons_weights(self, layer_idx, neuron_idx, weights):
        """
        Sets weights of selected neuron
        :param layer_idx - layer index
        :param neuron_idx - neuron index
        :param weights - list of weights (ints/floats)
        """
        self.layers[layer_idx][neuron_idx].weights = weights

    def get_neuron(self, layer_idx, neuron_idx):
        """
        Gets Neuron
        :param layer_idx - layer index
        :param neuron_idx - neuron index
        :return Neuron object
        """
        return self.layers[layer_idx][neuron_idx]

    def get_weights(self, layer_idx):
        """
        Gets weights of selected layer
        :param layer_idx - layer index
        :return weights - neurons weights as list of np.arrays
        """
        weights = list()
        for i in range(0, self.get_neurons_count(layer_idx)):
            neuron = self.get_neuron(layer_idx, i)
            weights.append(neuron.weights)
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
        for i in range(len(self.layers[0])):
            weights.append(self._get_neuron_weights(0, i))
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
        for neuron in self.layers[layer_idx]:
            y.append(neuron.compute(x))
        return np.array(y)
    
    @time_measure
    def run(self, x):
        """
        Run network
        :param x - input data as np.array
        :return y - output value as int/float
        """
        if len(x) != len(self.layers[0]):
            raise Exception("First layer has to have {0} neurons!".format(len(x)))

        self._print('Running network...')
        y_1 = self._run_first_layer(x)
        self._print('y_0:', y_1)
        y_i = y_1
        for i in range(len(self.layers) - 1):
            i += 1
            y_i = self._run_layer(y_i, i)
            self._print('y_{0}:'.format(i), y_i)
        return np.array(y_i)

    @time_measure
    def learn(self, x, z):
        """
        Learning with back propagation.
        :param x - input data as np.array
        :param z - desired output value as int/float
        """
        self._print('Learning network...')
        y = self.run(x)
        sigma = z - y
        
        # last layer:
        for neuron_idx in range(self.get_neurons_count(self.last_layer_idx())):
            self.layers[self.last_layer_idx()][neuron_idx].sigma = sigma
            self.layers[self.last_layer_idx()][neuron_idx].delta_w = self.learning_function(self.layers[self.last_layer_idx()][neuron_idx].last_x, y, sigma)

        for layer_idx in range(len(self.layers) - 2, -1, -1):
            for neuron_idx in range(len(self.layers[layer_idx])):
                sigm_sum = 0
                # sigma for n-th layer is a sum of sigma's from n+1 layer multiplied by proper weights from n+1 layer
                for neuron_idx_2 in range(len(self.layers[layer_idx + 1])):
                    sigm_sum += self.layers[layer_idx + 1][neuron_idx_2].sigma * self.layers[layer_idx + 1][neuron_idx_2].weights[neuron_idx]
                self.layers[layer_idx][neuron_idx].sigma = sigm_sum
                self.layers[layer_idx][neuron_idx].delta_w = self.learning_function(
                                    self.layers[layer_idx][neuron_idx].last_x,
                                    self.layers[layer_idx][neuron_idx].last_y, sigma)
                self.layers[layer_idx][neuron_idx].delta_bias = self.learning_function(
                                    1,
                                    self.layers[layer_idx][neuron_idx].bias, sigma)

        for layer_idx in range(len(self.layers) - 1, 0, -1):  # do not apply change to first layer
            for neuron_idx in range(len(self.layers[layer_idx])):
                self.layers[layer_idx][neuron_idx].apply_back_propagation()


def example():
    # init network - 2 inputs, 2 Neurons in hidden layer and 1 neuron for output
    net = NeuralNetwork(layer_size=[2, 2, 1])
    
    # setting weights:
    net.set_neurons_weights(1, 0, np.array(([0, 1]), dtype=float))
    net.set_neurons_weights(1, 1, np.array(([1, 0]), dtype=float))
    net.set_neurons_weights(2, 0, np.array(([1, 1]), dtype=float))

    x = np.array(([-1, 1]), dtype=float)
    print 'x:', x
    net.run(x)
    net.learn(x, np.array([0.5]))
    print net
    