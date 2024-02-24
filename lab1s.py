from math import exp
from random import random
import numpy as np

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, dataset):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.network = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)
        self.dataset = dataset

    # Calculate the derivative of an neuron output
    def sigmoid(self, output):
        return 1 / (1 + np.exp(-output))

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(neuron['output'] - expected[j])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.sigmoid(neuron['output'])

    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] -= l_rate * neuron['delta']

    # Train a network for a fixed number of epochs
    def train_network(self, l_rate, n_epoch):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in self.dataset:
                outputs = self.forward_propagate(self.network, row)
                expected = [0 for i in range(self.n_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward_propagate_error(self.network, expected)
                self.update_weights(self.network, row, l_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


    ############################
    # Calculate neuron activation for an input
    # зважена сума
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    # функція активації нейрона
    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation)) * 10

    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
        return output * (1.0 - output)

    # Forward propagate input to a network output
    # пряме поширенняю вихідне значення нейрона зберігається у neuron['output']
    # new_inputs - виходи з прошарку, входи до останнього нейрона
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Make a prediction with a network
    def predict(self, row):
        outputs = self.forward_propagate(self.network, row)
        return outputs.index(max(outputs))


# Test training backprop algorithm
# Test making predictions with the network
dataset = np.array([[2.54, 5.28, 0.78, 5.72],
                            [5.28, 0.78, 5.72, 0.58],
                            [0.78, 5.72, 0.58, 4.65],
                            [5.72, 0.58, 4.65, 0.91],
                            [0.58, 4.65, 0.91, 5.80],
                            [4.65, 0.91, 5.80,1.76],
                            [0.91, 5.80, 1.76, 5.67],
                            [5.80, 1.76, 5.67, 1.73],
                            [1.76, 5.67, 1.73, 5.70],
                            [5.67, 1.73, 5.70, 1.03]])

# network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
#  [{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
# for row in dataset:
#  prediction = predict(network, row)
#  print('Expected=%d, Got=%d' % (row[-1], prediction))

# test forward propagation
myNetwork = NeuralNetwork(3, 4, 1, dataset)
# myNetwork = NeuralNetwork(2, 2, 1, [[1, 1, 1], [0, 0, 0]])

# for layer in myNetwork.network:
#      print(layer)

# print(myNetwork.activate())

inputs = [2.54, 5.28, 0.78, 5.72]
for layer in myNetwork.network:
    new_inputs = []
    for neuron in layer:
        activation = myNetwork.activate(neuron['weights'], inputs)
        neuron['output'] = myNetwork.transfer(activation)
        new_inputs.append(neuron['output'])
    inputs = new_inputs
print(inputs)

# row = [3, 4, 1]
# output = forward_propagate(network, row)
# print(output)
