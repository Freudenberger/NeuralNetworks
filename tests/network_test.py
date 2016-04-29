import unittest
import numpy as np
import neural_networks


class Test(unittest.TestCase):

    def setUp(self):
        self.activation_function = lambda x: np.tanh(x)
        self.activation_fun_prime = lambda x: (1 - x) * (1 + x)

        self.net = neural_networks.NeuralNetwork(layer_size=[2, 2, 1],
                    activation_fun_prime=self.activation_fun_prime, activation_function=self.activation_function)

        # setting weights:
        self.net.set_neurons_weights(1, 0, np.array(([0, 1]), dtype=float))
        self.net.set_neurons_weights(1, 1, np.array(([1, 0]), dtype=float))
        self.net.set_neurons_weights(2, 0, np.array(([1, 1]), dtype=float))

        self.net.verbose = False
        self.net.eta = 2 * 0.05


    def tearDown(self):
        self.net = None


    def test_neural_simple_run(self):
        # Assign:
        x = np.array(([-1, 1]), dtype=float)
        y_2_expected = [0]
        y_1_expected = [ 0.76159416, -0.76159416]

        # Act:
        y_2_actual = self.net.run(x)
        y_1_actual = self.net.get_outputs(1)

        # Assert:
        msg = ""
        if np.allclose(y_1_actual, y_1_expected, rtol=1e-08, atol=1e-08) == False:
            msg += "Outputs not equal on first layer, expected: {0}, actual: {1}\n".format(y_1_expected, y_1_actual)
        if np.array_equal(y_2_actual, y_2_expected) == False:
            msg += "Outputs not equal on last layer, expected: {0}, actual: {1}\n".format(y_2_expected, y_2_actual)

        self.assertFalse(msg, msg)


    def test_neural_network_back_propagation_learning(self):
        # Assign:
        x = np.array(([-1, 1]), dtype=float)
        y_2_expected = [0.02435477]
        y_1_expected = [ 0.76159416, -0.76159416]
        w_2_expected = [ 1.0159925, 0.9840075]
        z = np.array([0.5])

        # Act:
        self.net.learn(x, z)
        y_2_actual = self.net.run(x)
        y_1_actual = self.net.get_outputs(1)
        w_2_actual = self.net.get_weights(2)

        # Assert:
        msg = ""
        if np.allclose(y_1_actual, y_1_expected, rtol=1e-08, atol=1e-08) == False:
            msg += "Outputs not equal on first layer, expected: {0}, actual: {1}\n".format(y_1_expected, y_1_actual)
        if np.allclose(y_2_actual, y_2_expected, rtol=1e-08, atol=1e-08) == False:
            msg += "Outputs not equal on last layer, expected: {0}, actual: {1}\n".format(y_2_expected, y_2_actual)
        if np.allclose(w_2_actual, w_2_expected, rtol=1e-08, atol=1e-08) == False:
            msg += "Weights not equal on last layer, expected: {0}, actual: {1}\n".format(w_2_expected, w_2_actual)

        self.assertFalse(msg, msg)


    def test_neural_network_hebb_learning(self):
        # Assign:
        x = np.array(([-1, 1]), dtype=float)
        y_2_expected = [0.31902101]
        y_1_expected = [0.88112962, -0.55057281]
        w_2_expected = [1.0, 1.0]

        # Act:
        self.net.hebb_learn(x, eta=0.5)
        y_2_actual = self.net.run(x)
        y_1_actual = self.net.get_outputs(1)
        w_2_actual = self.net.get_weights(2)

        # Assert:
        msg = ""
        if np.allclose(y_1_actual, y_1_expected, rtol=1e-08, atol=1e-08) == False:
            msg += "Outputs not equal on first layer, expected: {0}, actual: {1}\n".format(y_1_expected, y_1_actual)
        if np.allclose(y_2_actual, y_2_expected, rtol=1e-08, atol=1e-08) == False:
            msg += "Outputs not equal on last layer, expected: {0}, actual: {1}\n".format(y_2_expected, y_2_actual)
        if np.allclose(w_2_actual, w_2_expected, rtol=1e-08, atol=1e-08) == False:
            msg += "Weights not equal on last layer, expected: {0}, actual: {1}\n".format(w_2_expected, w_2_actual)

        self.assertFalse(msg, msg)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
