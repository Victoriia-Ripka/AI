# from math import exp
from random import random
import numpy as np

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, l_rate, dataset):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.l_rate = l_rate
        self.network = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden)]} for i in range(n_outputs)]
        self.network.append(output_layer)
        self.dataset = dataset


    # зважена сума
    def activate(self, weights, inputs):
        activation = 0
        for i in range(len(inputs)-1):
            activation += weights[i] * inputs[i]
        return activation


    # де тут множити на 10 і як 
    # функція активації нейрона - сигмоїд
    def transfer(self, activation):
        return 10.0  / (1.0 + np.exp(-activation))


    # похідна сигмоїд для обрахунку похибки
    def transfer_derivative(self, output):
        return 10.0 * np.exp(-output) / (1.0 + np.exp(-output)) ** 2
    

    # todo
    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            #  вихід
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            # прихований прошарок
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(neuron['output'] - expected)
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output']) * self.network[1][0]['weights'][j]
                # * neuton['input] 


    # todo
    # Update network weights with error
    def update_weights(self, row):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):

                    neuron['weights'][j] -= neuron['delta'] * self.l_rate * inputs[j] * neuron['weights'][j]
                neuron['weights'][-1] -= neuron['delta'] * self.l_rate


    # Train a network for a fixed number of epochs
    def train_network(self, n_epoch):
        for epoch in range(n_epoch):
            sum_error = 0
            expected = [row[-1] for row in self.dataset]

            for row, target in zip(self.dataset, expected):
                output = self.forward_propagate(row)
                # Yi – yi === обчислене значення виходу нейрона – правильне значення наступного члену часового ряду
                sum_error += (output - target) ** 2
                self.backward_propagate_error(target)
                self.update_weights(row)
            
            if epoch == 0 or (epoch + 1) % 100 == 0:
                print('[INFO] epoch=%d, lrate=%.1f, error=%.4f' % (epoch, self.l_rate, sum_error))


    # Forward propagate input to a network output
    # пряме поширення: вихідне значення нейрона зберігається у neuron['output']
    # new_inputs - виходи з прошарку, входи до останнього нейрона
    def forward_propagate(self, row):
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], row)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs[0]
    

    # Make a prediction with a network
    def predict(self, row):
        output = self.forward_propagate(row)
        return output


# Test training backprop algorithm
l_rate = 0.1
n_epoch = 20000
dataset = [[2.54, 5.28, 0.78, 5.72],
            [5.28, 0.78, 5.72, 0.58],
            [0.78, 5.72, 0.58, 4.65],
            [5.72, 0.58, 4.65, 0.91],
            [0.58, 4.65, 0.91, 5.80],
            [4.65, 0.91, 5.80, 1.76],
            [0.91, 5.80, 1.76, 5.67],
            [5.80, 1.76, 5.67, 1.73],
            [1.76, 5.67, 1.73, 5.70],
            [5.67, 1.73, 5.70, 1.03]]

myNetwork = NeuralNetwork(3, 4, 1, l_rate, dataset)

for layer in myNetwork.network:
    print(layer)

myNetwork.train_network(n_epoch)

for layer in myNetwork.network:
    print(layer)

# i = 0
# for row in dataset:
#     prediction = myNetwork.predict(dataset[i])
#     print('Expected=%.2f, Got=%.4f' % (row[-1], prediction))
#     i += 1

row1 = [1.73, 5.70, 1.03, 5.00]
row2 = [5.70, 1.03, 5.00, 1.79]
prediction1 = myNetwork.predict(row1)
print('Expected=%.2f, Got=%.4f' % (row1[-1], prediction1))
prediction2 = myNetwork.predict(row2)
print('Expected=%.2f, Got=%.4f' % (row2[-1], prediction2))