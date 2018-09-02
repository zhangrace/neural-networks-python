import numpy as np


class Network:
    """Represents a densely connected feed-forward neural network with ReLU
    activation functions.

    Performs vectorized versions of prediction and backpropagation.
    """
    def __init__(self, input_size, output_size, hidden_sizes, learning_rate=0.001,
                 weight_scale=1.0, converge=1e-2):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.layer_sizes) - 1
        self.activations = [np.zeros(s) for s in self.layer_sizes]
        limits = [np.sqrt(6 / (fan_in + fan_out)) for fan_in, fan_out in
                  zip(self.layer_sizes, self.layer_sizes[1:])]
        self.neuron_weights = [np.random.uniform(-l,l, [s2,s1]) for
                        l,s1,s2 in zip(limits, self.layer_sizes, self.layer_sizes[1:])]
        self.bias_weights = [np.zeros(s) for s in self.layer_sizes[1:]]
        self.deltas = [np.zeros(s) for s in self.layer_sizes[1:]]
        self.learning_rate = learning_rate
        self.converge = converge

    def predict(self, input_vector):
        """Computes the network's output for a given input_vector.

        input_vector: numpy array of input activations
        returns: numpy array of output activations
        """
        self.activations[0] = input_vector
        for a in range(1, len(self.activations)):
            v_weights = np.dot(self.neuron_weights[a-1],self.activations[a-1])
            v_weights += self.bias_weights[a-1]
            v_weights[v_weights<0]=0
            self.activations[a] = v_weights

        return self.activations[-1]

    def backpropagation(self, target_vector):
        """Updates all weights for a single step of stochastic gradient descent.

        Assumes that predict has just been called on the input vector
        corresponding to the given target_vector.

        target_vector: numpy array of expected outpur activations
        returns: nothing
        """
        old_weights = [i.copy() for i in self.neuron_weights]

        for a in range(len(self.activations)-1, 0, -1):
            relu_out = np.copy(self.activations[a])
            relu_out[relu_out<=0] = 0
            relu_out[relu_out>1] = 1
            if a == len(self.activations)-1:
                self.deltas[a-1] = relu_out*(target_vector - self.activations[a])
            else:
                self.deltas[a-1] = np.dot(self.deltas[a], old_weights[a]) * relu_out

            self.neuron_weights[a-1] = old_weights[a-1] + np.outer(self.deltas[a-1], self.activations[a-1]) * self.learning_rate
            self.bias_weights[a-1] += self.learning_rate * self.deltas[a-1]


    def train(self, data_set, epochs, verbose=False):
        """Runs repeated prediction & backpropagation steps to learn the data.

        data_set: a list of (input_vector, target_vector) pairs
        epochs: maximum number of times to loop through the data set.
        verbose: if False, nothing is printed; if True, prints the epoch on
                 on which training converged (or that it didn't)
        returns: nothing

        Runs iterations loops through the data set (shuffled each time).
        Each loop runs predict to compute activations, then backpropagation
        to update weights on every example in the data set. If all outputs
        are within self.converge of their targets (self.test returns 1.0)
        training stops regardless of the iteration.
        """
        for i in range(epochs):
            np.random.shuffle(data_set)

            for data in data_set:

                self.predict(data[0])
                self.backpropagation(data[1])

            if (self.test(data_set) == 1):
                if verbose:
                    print("epoch converged: " + str(i))
                return
        if verbose:
            print("did not converge")

    def test(self, data_set):
        """Predicts every input in the data set and returns the accuracy.

        data_set: a list of (input_vector, target_vector) pairs
        returns: accuracy, the fraction of output vectors within self.converge
                 of their targets (as measured by numpy.allclose's rtol).

        Calls predict() on each input vector in the data set, and compares the
        result to the corresponding target vector from the data set.
        """
        num_converged = 0.0
        for input_vector, target_vector in data_set:
            if np.allclose(self.predict(input_vector), target_vector, rtol=self.converge):
                num_converged+=1
        return num_converged/len(data_set)

    def categorical_accuracy(self, data_set):
        correct = 0
        for input_vector, target_vector in data_set:
            predictions = self.predict(input_vector)
            if np.argmax(predictions) == np.argmax(target_vector):
                correct += 1
        return correct / len(data_set)

    def __repr__(self):
        s = "Neural Network\n"
        s += "layer 0:\n  " + str(self.layer_sizes[0]) + " input nodes\n"
        for i in range(len(self.layer_sizes) - 1):
            s += "layer " + str(i+1) + ":\n"
            s += "  neuron weights:\n"
            s += str(self.neuron_weights[i]) + "\n"
            s += "  bias weights:\n"
            s += str(self.bias_weights[i]) + "\n"
        return s

def train_XOR():
    # train a network with one hidden layer on XOR
    data_set = [(np.array([0,0]), np.array([0.0])),
                (np.array([0,1]), np.array([1.0])),
                (np.array([1,0]), np.array([1.0])),
                (np.array([1,1]), np.array([0.0]))]
    nn = Network(2, 1, [2], learning_rate=0.01, weight_scale=1.0)
    nn.train(data_set, 10000, verbose=True)
    print("\nresulting", nn, "\n")
    print("accuracy =", nn.test(data_set))
    for input_vector, target_vector in data_set:
        output_vector = nn.predict(input_vector)
        print("input:", input_vector, "target:", target_vector, "output:", output_vector)

def train_MNIST():
    print()
    data = np.load("mnist.npz")
    nn = Network(784,10,[200,100], learning_rate=0.01)
    training_data = list(zip(data["x_train"], data["y_train"]))
    nn.train(training_data, 5)
    testing_data = list(zip(data["x_test"], data["y_test"]))
    print("MNIST test accuracy:", nn.test(testing_data))
    print("MNIST categorical accuracy:", nn.categorical_accuracy(testing_data))

if __name__ == "__main__":
    train_XOR()
    train_MNIST()
