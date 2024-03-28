import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, function):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_function = self.initialize_activation_func(function)

        self.weights = []
        self.biases = []

        self.initialize_layer(self.input_size, self.hidden_layers[0])
        for i in range(1, len(self.hidden_layers)):

            if i == len(self.hidden_layers) - 1:
                layers = self.output_size 
            else:
                layers = self.hidden_layers[i]
            
            self.initialize_layer(self.hidden_layers[i - 1], layers)


    def initialize_layer(self, input_size, output_size):
        self.weights.append(np.random.randn(input_size, output_size)) 
        self.biases.append(np.zeros((1, output_size)))
    

    def initialize_activation_func(self, function_name):
        activation_functions = {
            'relu': self.relu,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh
        }
        return activation_functions.get(function_name)
    

    @staticmethod
    def relu(x, derive=False):
        if not derive:
            return np.maximum(0, x)
        return np.where(x >= 0, 1, 0)


    @staticmethod
    def sigmoid(x, derive=False):
        clipped_x = np.clip(x, -500, 500)
        if not derive:
            return 1 / (1 + np.exp(-clipped_x))
        return x * (1 - x)


    @staticmethod
    def tanh(x, derive=False):
        if not derive:
            return np.tanh(x)
        return 1 - np.square(np.tanh(x))


    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Додання зміщення для стабільності
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


    def predict(self, x):
        return np.argmax(self.forward_propagate(x), axis=1)


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


    def backward_propagate_error(self, data, codes, l_rate):
        num_samples = data.shape[0]

        # обрахування похибки для прихованого прошарку
        output_error = self.layer_outputs[-1] - codes
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


    def train_network(self, data, codes, n_epoch, l_rate, error_threshold=0.0001):
        for epoch in range(n_epoch):
            output = self.forward_propagate(data)
            self.layer_outputs = self.layer_inputs.copy()
            self.layer_outputs.append(output)
            self.backward_propagate_error(data, codes, l_rate)

            if epoch % 5000 == 0:
                error = -np.mean(np.log(output[np.arange(len(codes)), np.argmax(codes, axis = 1)]))
                print('[INFO] epoch=%d, error=%.4f' % (epoch, error))

                if error < error_threshold:
                    print('[INFO] Training stopped. Error is below the threshold.')
                    break



data = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

test_data1 = np.array([[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

test_data2 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

test_data3 = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

codes = np.array([0, 1, 2, 3, 4])



l_rate = 0.05
n_epoch = 25000
input_layer_size = data.shape[1]
hidden_layers = [36, 1]
output_layer_size = len(codes)



print("\nSigmoid activation function\n")
nn_S = NeuralNetwork(input_layer_size, hidden_layers, output_layer_size, 'sigmoid')
nn_S.train_network(data, np.eye(output_layer_size)[codes], n_epoch, l_rate)
print('\nExpected result:', codes)
print('Predictions:    ', nn_S.predict(test_data2), '\n')

print("\nTanh activation function\n")
nn_T = NeuralNetwork(input_layer_size, hidden_layers, output_layer_size, 'tanh')
nn_T.train_network(data, np.eye(output_layer_size)[codes], n_epoch, l_rate)
print('\nExpected result:', codes)
print('Predictions:    ', nn_T.predict(test_data2), '\n')

print("\nRelu activation function\n")
nn_R = NeuralNetwork(input_layer_size, hidden_layers, output_layer_size, 'relu')
nn_R.train_network(data, np.eye(output_layer_size)[codes], n_epoch, l_rate)
print('\nExpected result:', codes)
print('Predictions:    ', nn_R.predict(test_data2), '\n')

