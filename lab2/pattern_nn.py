import numpy as np

# функції активації та їх похідні
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.power(tanh(x), 2)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Додання зміщення для стабільності
    return exp_x / np.sum(exp_x, axis=0)

class NeuralNetwork:
    def __init__(self, n_hidden, l_rate, n_epoch, filename, f, f_derivative):
        self.l_rate = l_rate
        self.n_epoch = n_epoch

        count_of_data, n_inputs, n_outputs, data, codes = self.read_dataset(filename)

        self.dataset_inputs = data
        self.dataset_outputs = codes
        self.weights_hidden_input = np.random.randn(n_inputs, n_hidden)
        self.hidden_bias = np.zeros((1, n_hidden))
        self.weights_hidden_output = np.random.randn(n_hidden, n_outputs)
        self.output_bias = np.zeros((1, n_outputs))


    def read_dataset(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        count_of_data = int(lines[0])
        count_of_inputs = int(lines[1])
        count_of_outputs = int(lines[2])
        data = []
        codes = []

        for i in range(4, len(lines), 3):
            data_line = lines[i].strip()
            code_line = lines[i + 1].strip()

            data.append([x for x in data_line.split(',')])
            codes.append(code_line)

        return count_of_data, count_of_inputs, count_of_outputs, data, codes


    def forward_propagate(self, data):
        # вираховується зважена сума до прошарку
        self.hidden_input = np.dot(data, self.weights_hidden_input) + self.hidden_bias
        # вихід після функції активації
        self.hidden_output = self.transfer(self.hidden_input)
        # вираховується вихід з прошарку
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.output_bias
        return self.output
    

    def backward_propagate_error(self, output):
        # обрахування похибки для прихованого прошарку
        d_output = np.subtract(self.dataset_outputs, output)
        d_hidden_output = np.dot(d_output, self.weights_hidden_output.T)
        d_hidden_input = d_hidden_output * self.transfer_derivative(self.hidden_output)
        self.update_weights(d_output, d_hidden_input)
    

    def update_weights(self, d_output, d_hidden_input):
        # оновлення вагів з входів до прихованого прошарку
        self.weights_hidden_input += np.dot(self.dataset_inputs.T, d_hidden_input) * self.l_rate
        self.hidden_bias += np.sum(d_hidden_input, axis=0, keepdims=True) * self.l_rate

        # оновлення вагів з прихованого прошарку до виходу
        self.weights_hidden_output += np.dot(self.hidden_output.T, d_output) * self.l_rate
        self.output_bias += np.sum(d_output, axis=0, keepdims=True) * self.l_rate


    def train_network(self, error_threshold=0.00001):
        for epoch in range(self.n_epoch):
            output = self.forward_propagate(self.dataset_inputs)
            self.backward_propagate_error(output)

            current_error = abs(float(output[0][0]) - float(self.dataset_outputs[0][0]))
            if epoch % 100 == 0:
                print('[INFO] epoch=%d, error=%.4f' % (epoch, current_error))

            if current_error < error_threshold:
                print('[INFO] Training stopped. Error is below the threshold.')
                break



l_rate = 0.08
n_epoch = 15000
filename = "train.py"

my_nn = NeuralNetwork(36, l_rate, n_epoch, filename, sigmoid, sigmoid_derivative)
# без прихованого шару
# 1 прихований шар 36 нейронів
# 1 прихований шар 72 нейрони
# 2 приховані шари по 36 нейронів
            
# функції активації: ReLU, Sigmoid, Tanh, Softmax (на виході)