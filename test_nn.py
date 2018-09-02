#! /usr/bin/env python3

import unittest
import vectorized_nn as NN
import numpy as np

class ApproximateTester(unittest.TestCase):
    def assertClose(self, actual, expected, rtol=1e-8, message=""):
        self.assertTrue(np.allclose(actual, expected, rtol),
                        (message + "\nactual={} and expected={} are not " +
                        "within tolerance={}").format(actual, expected, rtol))

class Test_Predict_2332(ApproximateTester):
    def setUp(self):
        self.nn = NN.Network(2,2,[3,3])
        self.nn.neuron_weights[0][:,0] = np.arange(0.5,2,0.5)
        self.nn.neuron_weights[0][:,1] = np.arange(0.5,-1,-0.5)
        self.nn.neuron_weights[1][0] = np.arange(3)
        self.nn.neuron_weights[1][1] = np.arange(0,-3,-1)
        self.nn.neuron_weights[1][2] = np.arange(-1,2)
        self.nn.neuron_weights[2][0,:] = np.arange(1,-2,-1)
        self.nn.neuron_weights[2][1,:] = np.arange(2,0.5,-0.5)
        self.nn.bias_weights[0] = np.ones(3)
        self.nn.bias_weights[1] = -np.ones(3)
        self.nn.bias_weights[2] = np.zeros(2)

    def test_predict_layer0(self):
        a0 = np.array([1.0, 0.5])
        self.nn.predict(a0)
        self.assertClose(self.nn.activations[0], a0, message="Incorrect activations for input layer.")

    def test_predict_layer1(self):
        a0 = np.array([1.0, 0.5])
        a1 = np.array([1.75, 2.0, 2.25])
        self.nn.predict(a0)
        self.assertClose(self.nn.activations[1], a1, message="Incorrect activations for hidden layer 1.")

    def test_predict_layer2(self):
        a0 = np.array([1.0, 0.5])
        a2 = np.array([5.5, 0.0, 0.0])
        self.nn.predict(a0)
        self.assertClose(self.nn.activations[2], a2, message="Incorrect activations for hidden layer 2.")

    def test_predict_layer3(self):
        a0 = np.array([1.0, 0.5])
        a3 = np.array([5.5, 11.])
        self.nn.predict(a0)
        self.assertClose(self.nn.activations[3], a3, message="Incorrect activations for output layer.")


class Test_Backprop_2332(ApproximateTester):
    def setUp(self):
        self.nn = NN.Network(2,2,[3,3], learning_rate=0.01)
        self.nn.neuron_weights[0][:,0] = np.arange(0.5,2,0.5)
        self.nn.neuron_weights[0][:,1] = np.arange(0.5,-1,-0.5)
        self.nn.neuron_weights[1][0] = np.arange(3)
        self.nn.neuron_weights[1][1] = np.arange(0,-3,-1)
        self.nn.neuron_weights[1][2] = np.arange(-1,2)
        self.nn.neuron_weights[2][0,:] = np.arange(1,-2,-1)
        self.nn.neuron_weights[2][1,:] = np.arange(2,0.5,-0.5)
        self.nn.bias_weights[0] = np.ones(3)
        self.nn.bias_weights[1] = -np.ones(3)
        self.nn.bias_weights[2] = np.zeros(2)

    def test_backprop_deltas_layer3(self):
        a0 = np.array([1.0, 0.5])
        t = np.array([0.0, 0.5])
        d3 = np.array([-5.5, -10.5])
        self.nn.predict(a0)
        self.nn.backpropagation(t)
        self.assertClose(self.nn.deltas[2], d3, message="Incorrect deltas for output layer.")

    def test_backprop_deltas_layer2(self):
        a0 = np.array([1.0, 0.5])
        t = np.array([0.0, 0.5])
        d2 = np.array([-26.5, 0.0, 0.0])
        self.nn.predict(a0)
        self.nn.backpropagation(t)
        self.assertClose(self.nn.deltas[1], d2, message="Incorrect deltas for hidden layer 2.")

    def test_backprop_deltas_layer1(self):
        a0 = np.array([1.0, 0.5])
        t = np.array([0.0, 0.5])
        d1 = np.array([0.0, -26.5, -53.0])
        self.nn.predict(a0)
        self.nn.backpropagation(t)
        self.assertClose(self.nn.deltas[0], d1, message="Incorrect deltas for hidden layer 1.")

    def test_backprop_weights_layer3(self):
        a0 = np.array([1.0, 0.5])
        t = np.array([0.0, 0.5])
        w3 = np.array([[0.6975, 0.0, -1.0], [1.4225, 1.5, 1.0]])
        self.nn.predict(a0)
        self.nn.backpropagation(t)
        self.assertClose(self.nn.neuron_weights[2], w3, message="Incorrect neuron weights for output layer.")

    def test_backprop_weights_layer2(self):
        a0 = np.array([1.0, 0.5])
        t = np.array([0.0, 0.5])
        w2 = np.array([[-0.46375, 0.47, 1.40375], [0.0, -1.0, -2.0], [-1.0, 0.0, 1.0]])
        self.nn.predict(a0)
        self.nn.backpropagation(t)
        self.assertClose(self.nn.neuron_weights[1], w2, message="Incorrect neuron weights for hidden layer 2.")

    def test_backprop_weights_layer1(self):
        a0 = np.array([1.0, 0.5])
        t = np.array([0.0, 0.5])
        w1 = np.array([[0.5, 0.5], [0.735, -0.1325], [0.97, -0.765]])
        self.nn.predict(a0)
        self.nn.backpropagation(t)
        self.assertClose(self.nn.neuron_weights[0], w1, message="Incorrect neuron weights for hidden layer 1.")

    def test_backprop_biases_layer3(self):
        a0 = np.array([1.0, 0.5])
        t = np.array([0.0, 0.5])
        b3 = np.array([-0.055, -0.105])
        self.nn.predict(a0)
        self.nn.backpropagation(t)
        self.assertClose(self.nn.bias_weights[2], b3, message="Incorrect bias weights for output layer.")

    def test_backprop_biases_layer2(self):
        a0 = np.array([1.0, 0.5])
        t = np.array([0.0, 0.5])
        b2 = np.array([-1.265, -1.0, -1.0])
        self.nn.predict(a0)
        self.nn.backpropagation(t)
        self.assertClose(self.nn.bias_weights[1], b2, message="Incorrect bias weights for hidden layer 2.")

    def test_backprop_biases_layer1(self):
        a0 = np.array([1.0, 0.5])
        t = np.array([0.0, 0.5])
        b1 = np.array([1.0, 0.735, 0.47])
        self.nn.predict(a0)
        self.nn.backpropagation(t)
        self.assertClose(self.nn.bias_weights[0], b1, message="Incorrect bias weights for hidden layer 1.")


if __name__ == "__main__":
    unittest.main()
