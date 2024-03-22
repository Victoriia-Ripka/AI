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
    # n_hidden_layer, n_hidden_neurons
    def __init__(self, n_hidden, l_rate, n_epoch, filename, f, f_derivative):
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.f = f
        self.f_derivative = f_derivative

        n_inputs, n_outputs, data, codes = self.read_dataset(filename)

        self.dataset_inputs = data
        self.dataset_outputs = codes
        self.weights_hidden_input = np.random.randn(n_inputs, n_hidden)
        self.hidden_bias = np.zeros((1, n_hidden))
        self.weights_hidden_output = np.random.randn(n_hidden, n_outputs)
        self.output_bias = np.zeros((1, n_outputs))


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

        return count_of_inputs, count_of_outputs, data, codes


    def forward_propagate(self):
        output = []
        
        for i in range(len(self.dataset_inputs)):
            # вираховується зважена сума до прошарку
            data_numeric = np.array(self.dataset_inputs[i], dtype=int)
            self.hidden_input = np.dot(data_numeric, self.weights_hidden_input) + self.hidden_bias[0]
            # вихід після функції активації
            self.hidden_output = self.f(self.hidden_input)
            # вираховується вихід з прошарку
            result = np.dot(self.hidden_output, self.weights_hidden_output) + self.output_bias
            output.append(result[0][0])
        return output


    def backward_propagate_error(self, output):
        print(output)
        # обрахування похибки для прихованого прошарку
        output_numeric = np.array(self.dataset_outputs, dtype=int)
        d_output = np.subtract(output_numeric, output)

        print(d_output)
        print(self.weights_hidden_output.T)
        d_hidden_output = np.dot(d_output, self.weights_hidden_output.T)
        d_hidden_input = d_hidden_output * self.f_derivative(self.hidden_output)
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
            output = self.forward_propagate()
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
my_nn.train_network()
# без прихованого шару
# 1 прихований шар 36 нейронів
# 1 прихований шар 72 нейрони
# 2 приховані шари по 36 нейронів
            
# функції активації: ReLU, Sigmoid, Tanh, Softmax (на виході)