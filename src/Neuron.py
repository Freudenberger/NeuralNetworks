import numpy as np

class Neuron(object):
    def __init__(self, inputs, weights=None, activation_function=None):
        self.weights = weights
        self.activation_function = activation_function
        self.sigma = 0
        self.bias = 0
        self.delta_w = 0
        self.delta_bias = 0
        self.last_x = 0
        self.last_y = 0

        if weights is None:
            self.weights = np.random.rand(inputs)

        if activation_function is None:
            self.activation_function = lambda x: np.tanh(x)

    def __repr__(self):
        return "[Neuron] weights: {0}, sigma:{1}, delta_w:{2}, bias:{3}".format(self.weights, self.sigma, self.delta_w, self.bias)

    def compute(self, x):
        self.last_x = x
        y = np.dot(x, np.array((self.weights)))
        y = np.asscalar(y + self.bias)
        self.last_y = self.activation_function(y)
        return self.last_y

    def apply_back_propagation(self):
        self.weights += self.delta_w
        self.bias += self.delta_bias
