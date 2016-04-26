import numpy as np

class BaseNeuron(object):
    def __init__(self):
        pass
    
    def compute(self, x):
        raise NotImplementedError()
    

class SimpleNeuron(BaseNeuron):
    def __init__(self, input_size, weights=None, activation_function=None):
        """
        :param input_size - input size
        :param weights (optional) - list of weights; must be the size of inputs; if none then set to rand()
        :param activation_function (optional) - lambda representing activation function; if None then set to np.tanh(x)
        """
        super(BaseNeuron, self).__init__()
        self._input_size = input_size
        self.weights = weights
        self.activation_function = activation_function
        self.sigma = 0
        self.bias = 0
        self.delta_w = 0
        self.delta_bias = 0
        self.last_x = 0
        self.last_y = 0

        if weights is None:
            self.weights = np.random.rand(self._input_size)

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
