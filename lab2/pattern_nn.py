import numpy as np

# функції активації та їх похідні
def relu(x, derive=False):
    if not derive:
        return np.maximum(0, x)
    return np.where(x >= 0, 1, 0)

def sigmoid(x, derive=False):
    clipped_x = np.clip(x, -500, 500)
    if not derive:
        return 1 / (1 + np.exp(-clipped_x))
    return x * (1 - x)

def tanh(x, derive=False):
    if not derive:
        return np.tanh(x)
    return 1 - np.square(np.tanh(x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Додання зміщення для стабільності
    return exp_x / np.sum(exp_x, axis=0)

class NeuralNetwork:
    def __init__(self, n_hidden, filename, function):
        self.activation_function = function

        n_inputs, n_outputs, data, codes = self.read_dataset(filename)

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.dataset_inputs = data
        self.dataset_outputs = codes

        self.weights = []
        self.biases = []

        print(self.n_inputs, self.n_hidden)
        self.initialize_layer(self.n_inputs, self.n_hidden[0])
        for i in range(len(self.n_hidden)):

            if i == len(self.n_hidden) - 1:
                layers = self.n_outputs 
            else:
                self.n_hidden[i]
            
            self.initialize_layer(self.n_hidden[i - 1], layers)


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

            data.append([x for x in data_line.split(',')])
            codes.append(code_line)

        return count_of_inputs, count_of_outputs, np.array(data), np.array(codes)


    def initialize_layer(self, n_inputs, n_outputs):
        print(type(n_inputs), type(n_outputs))
        self.weights.append(np.random.randn(n_inputs, n_outputs)) 
        self.biases.append(np.zeros((1, n_outputs)))
    

    def predict(self, x):
        return np.argmax(self.forward_propagate(x), axis=1)


    def forward_propagate(self, data):
        input_layer = data
        self.layer_inputs = [input_layer]

        for i in range(len(self.weights)-1):
            # вираховується зважена сума до прошарків
            # data_numeric = np.array(self.dataset_inputs[i], dtype=int)
            output_layer = np.dot(input_layer, self.weights[i]) + self.biases[i]
            input_layer = self.activation_function(output_layer)
            self.layer_inputs.append(input_layer)
        
        # вихід з останнього прихованого шару з функцією активації softmax
        output = np.dot(input_layer, self.weights[-1]) + self.biases[-1]
        output = softmax(output)
        return output


    def backward_propagate_error(self, X, y, l_rate):
        num_samples = X.shape[0]

        # обрахування похибки для прихованого прошарку
        output_error = self.layer_output[-1] - y
        output_delta = output_error/num_samples
        self.weights[-1] -= l_rate * np.dot(self.layer_inputs[-1].T, output_delta)
        self.biases[-1] -= l_rate * np.sum(output_delta, axis=0, keepdims=True) 

        # 
        for i in range(len(self.weights) - 1):
            error = np.dot(output_delta, self.weights[i + 1].T)
            delta = error * self.activation_function(self.layer_inputs[i], True)
            self.weights[i] -= l_rate * np.dot(self.layer_inputs[i - 1].T, delta)
            self.biases[i] -= l_rate * np.sum(delta, axis=0, keepdims=True) 
            output_delta = delta

        # 
        error = np.dot(output_delta, self.weights[1].T)
        delta = error * self.activation_function(self.layer_inputs[0], True)
        self.weights[0] -= l_rate * np.dot(self.layer_inputs[0].T, output_delta)
        self.biases[0] -= l_rate * np.sum(output_delta, axis=0, keepdims=True) 


    def train_network(self, X, y, n_epoch, l_rate, error_threshold=0.00001):
        for epoch in range(n_epoch):
            output = self.forward_propagate(X)
            self.layer_outputs = self.layer_inputs.copy()
            self.layer_outputs.append(output)
            self.backward_propagate_error(X, y, l_rate)

            # Calculate the current error
            if epoch % 200 == 0:
                error = -np.mean(np.log(output[np.arange(len(y)), np.argmax(y, axis=1)]))
                print('[INFO] epoch=%d, error=%.4f' % (epoch, error))

                if error < error_threshold:
                    print('[INFO] Training stopped. Error is below the threshold.')
                    break


l_rate = 0.08
n_epoch = 5000
filename = "train.py"

# my_nn0S = NeuralNetwork([0], filename, 'sigmoid')
# my_nn0S.train_network()
# _, _, data, codes = my_nn0S.read_dataset('test1.py')
# print('prediction: ', my_nn0S.forward_propagate(data), '\n')

my_nn36S = NeuralNetwork([36], filename, 'sigmoid')
my_nn36S.train_network()
_, _, data, codes = my_nn36S.read_dataset('test1.py')
print('prediction: ', my_nn36S.forward_propagate(data), '\n')

# my_nn36x2S = NeuralNetwork([36, 36], filename, 'sigmoid')
# my_nn36S.train_network()
# _, _, data, codes = my_nn36S.read_dataset('test1.py')
# print('prediction: ', my_nn36S.forward_propagate(data), '\n')

# my_nn72S = NeuralNetwork([72], filename, 'sigmoid')
# my_nn72S.train_network()
# _, _, data, codes = my_nn72S.read_dataset('test1.py')
# print(my_nn72S.forward_propagate(data), '\n')

# my_nn0T = NeuralNetwork([0], filename, 'tanh')
# my_nn0T.train_network()
# _, _, data, codes = my_nn0T.read_dataset('test1.py')
# print(my_nn0T.forward_propagate(data), '\n')

# my_nn36T = NeuralNetwork([36], filename, 'tanh')
# my_nn36T.train_network()
# _, _, data, codes = my_nn36T.read_dataset('test1.py')
# print(my_nn36T.forward_propagate(data), '\n')

# my_nn36x2T = NeuralNetwork([36, 36], filename, 'tanh')
# my_nn36x2T.train_network()
# _, _, data, codes = my_nn36T.read_dataset('test1.py')
# print(my_nn36T.forward_propagate(data), '\n')

# my_nn72T = NeuralNetwork([72], filename, 'tanh')
# my_nn72T.train_network()
# _, _, data, codes = my_nn72T.read_dataset('test1.py')
# print(my_nn72T.forward_propagate(data), '\n')

# my_nn0R = NeuralNetwork([0], filename, 'relu')
# my_nn0R.train_network()
# _, _, data, codes = my_nn0R.read_dataset('test1.py')
# print(my_nn0R.forward_propagate(data), '\n')

# my_nn36R = NeuralNetwork([36], filename, 'relu')
# my_nn36R.train_network()
# _, _, data, codes = my_nn36R.read_dataset('test1.py')
# print(my_nn36R.forward_propagate(data), '\n')

# my_nn36x2R = NeuralNetwork([36, 36], filename, 'relu')
# my_nn36x2R.train_network()
# _, _, data, codes = my_nn36R.read_dataset('test1.py')
# print(my_nn36R.forward_propagate(data), '\n')

# my_nn72R = NeuralNetwork([72], filename, 'relu')
# my_nn72R.train_network()
# _, _, data, codes = my_nn72R.read_dataset('test1.py')
# print(my_nn72R.forward_propagate(data), '\n')


# без прихованого шару
# 1 прихований шар 36 нейронів
# 1 прихований шар 72 нейрони
# 2 приховані шари по 36 нейронів
            
# функції активації: ReLU, Sigmoid, Tanh, Softmax (на виході)