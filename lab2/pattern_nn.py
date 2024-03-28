import numpy as np


class NeuralNetwork:
    def __init__(self, hidden_layers, filename, function):
        input_size, output_size, data, codes = self.read_dataset(filename)

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dataset_inputs = data
        self.dataset_outputs = codes
        self.activation_function = self.initialize_activation_function(function)

        self.weights = []
        self.biases = []

        self.initialize_layer(self.input_size, self.hidden_layers[0])
        for i in range(1, len(self.hidden_layers)):

            if i == len(self.hidden_layers) - 1:
                layers = self.output_size 
            else:
                layers = self.hidden_layers[i]
            
            self.initialize_layer(self.hidden_layers[i - 1], layers)


    def read_dataset(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        count_of_inputs = int(lines[0])
        count_of_outputs = int(lines[1])
        data = []
        codes = []

        for i in range(3, len(lines), 3):
            data_line = lines[i].strip()
            code_line = lines[i + 1].strip()

            data.append([int(x) for x in data_line.split(',')])
            codes.append(int(code_line))

        return count_of_inputs, count_of_outputs, np.array(data), np.array(codes)


    def initialize_layer(self, input_size, output_size):
        self.weights.append(np.random.randn(input_size, output_size)) 
        self.biases.append(np.zeros((1, output_size)))
    

    def initialize_activation_function(self, function_name):
        activation_functions = {
            'relu': self.relu,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh
        }
        return activation_functions.get(function_name, self.sigmoid)
    

    def relu(self, x, derive=False):
        if not derive:
            return np.maximum(0, x)
        return np.where(x >= 0, 1, 0)


    def sigmoid(self, x, derive=False):
        clipped_x = np.clip(x, -500, 500)
        if not derive:
            return 1 / (1 + np.exp(-clipped_x))
        return x * (1 - x)


    def tanh(self, x, derive=False):
        if not derive:
            return np.tanh(x)
        return 1 - np.square(np.tanh(x))


    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Додання зміщення для стабільності
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


    def predict(self, x):
        return np.argmax(self.forward_propagate(x), axis=1)
        # return self.forward_propagate(x)


    def forward_propagate(self, data):
        layer_input = data
        self.layer_inputs = [layer_input]

        n_hidden_layers = len(self.weights) - 1
        for i in range(n_hidden_layers):
            # вираховується зважена сума до прошарків
            layer_output = np.dot(layer_input, self.weights[i]) + self.biases[i]
            layer_input = self.activation_function(layer_output)
            self.layer_inputs.append(layer_input)
        
        # вихід з останнього прихованого шару з функцією активації softmax
        output = np.dot(layer_input, self.weights[-1]) + self.biases[-1]
        # output = np.clip(output, -700, 700)
        output = self.softmax(output)
        return output


    def backward_propagate_error(self, l_rate):
        num_samples = self.dataset_inputs.shape[0]

        # обрахування похибки для прихованого прошарку
        output_error = self.layer_outputs[-1] - self.dataset_outputs
        output_delta = output_error/num_samples
        self.weights[-1] -= l_rate * np.dot(self.layer_inputs[-1].T, output_delta)
        self.biases[-1] -= l_rate * np.sum(output_delta, axis=0, keepdims=True) 

        # 
        for i in range(len(self.weights) - 2, 0, -1):        
            error = np.dot(output_delta, self.weights[i + 1].T)
            delta = error * self.activation_function(self.layer_inputs[i], True)
            self.weights[i] -= l_rate * np.dot(self.layer_inputs[i - 1].T, delta)
            self.biases[i] -= l_rate * np.sum(delta, axis=0, keepdims=True) 
            output_delta = delta

        # 
        error = np.dot(output_delta, self.weights[1].T)
        delta = error * self.activation_function(self.layer_inputs[0], True)
        self.weights[0] -= l_rate * np.dot(self.layer_inputs[0].T, delta)
        self.biases[0] -= l_rate * np.sum(delta, axis=0, keepdims=True) 


    def train_network(self, n_epoch, l_rate, error_threshold=0.0001):
        for epoch in range(n_epoch):
            output = self.forward_propagate(self.dataset_inputs)
            self.layer_outputs = self.layer_inputs.copy()
            self.layer_outputs.append(output)
            self.backward_propagate_error(l_rate)

            if epoch % 2000 == 0:
                error = -np.mean(np.log(output[np.arange(len(self.dataset_outputs)), np.argmax(self.dataset_outputs)]))
                print('[INFO] epoch=%d, error=%.4f' % (epoch, error))

                if error < error_threshold:
                    print('[INFO] Training stopped. Error is below the threshold.')
                    break


l_rate = 0.05
n_epoch = 15000
train_filename = "train.py"
test_filename = 'test1.py'
# hidden_layers = [0, 0]
# hidden_layers = [36, 1]
hidden_layers = [36, 2]
# hidden_layers = [72, 1]

print("\nSigmoid activation function\n")
my_nnS = NeuralNetwork(hidden_layers, train_filename, 'sigmoid')
my_nnS.train_network(n_epoch, l_rate)
_, _, data, _ = my_nnS.read_dataset(test_filename)
print('\nExpected result:', my_nnS.dataset_outputs)
print('Predictions:    ', my_nnS.predict(data), '\n')

print("\nTanh activation function\n")
my_nnT = NeuralNetwork(hidden_layers, train_filename, 'tanh')
my_nnT.train_network(n_epoch, l_rate, error_threshold=0.00005)
_, _, data, _ = my_nnT.read_dataset(test_filename)
print('\nExpected result:', my_nnT.dataset_outputs)
print('Predictions:    ', my_nnT.predict(data), '\n')

print("\nRelu activation function\n")
my_nnR = NeuralNetwork(hidden_layers, train_filename, 'relu')
my_nnR.train_network(n_epoch, l_rate)
_, _, data, _ = my_nnR.read_dataset(test_filename)
print('\nExpected result:', my_nnR.dataset_outputs)
print('Predictions:    ', my_nnR.predict(data), '\n')













# без прихованого шару
# 1 прихований шар 36 нейронів
# 1 прихований шар 72 нейрони
# 2 приховані шари по 36 нейронів
            
# функції активації: ReLU, Sigmoid, Tanh, Softmax (на виході)