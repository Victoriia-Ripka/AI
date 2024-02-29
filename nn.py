import numpy as np

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, l_rate, datasetX, datasetY, n_epoch):
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.dataset_inputs = datasetX
        self.dataset_outputs = datasetY
        self.weights_input_hidden = np.random.randn(n_inputs, n_hidden)
        self.hidden_bias = np.zeros((1, n_hidden))
        self.weights_output_hidden = np.random.randn(n_hidden, n_outputs)
        self.output_bias = np.zeros((1, n_outputs))


    def transfer(self, x):
        x = np.clip(x, -500, 500)
        return 1.0  / (1.0 + np.exp(-x))


    def transfer_derivative(self, output):
        return self.transfer(output) * ( 1.0 - self.transfer(output))
    

    def forward_propagate(self, data):
        self.hidden_input = np.dot(data, self.weights_input_hidden) + self.hidden_bias
        self.hidden_output = self.transfer(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_output_hidden) + self.output_bias
        return self.output
    

    def backward_propagate_error(self, output):
        d_output = np.subtract(self.dataset_outputs, output)
        d_hidden_output = np.dot(d_output, self.weights_output_hidden.T)
        d_hidden_input = d_hidden_output * self.transfer_derivative(self.hidden_output)
        self.update_weights(d_output, d_hidden_input)
    

    def update_weights(self, d_output, d_hidden_input):
        self.weights_output_hidden += np.dot(self.hidden_output.T, d_output) * self.l_rate
        self.output_bias += np.sum(d_output, axis=0, keepdims=True) * self.l_rate
        self.weights_input_hidden += np.dot(self.dataset_inputs.T, d_hidden_input) * self.l_rate
        self.hidden_bias += np.sum(d_hidden_input, axis=0, keepdims=True) * self.l_rate


    def train_network(self, error_threshold=0.0001):
        for epoch in range(self.n_epoch):
            output = self.forward_propagate(self.dataset_inputs)
            self.backward_propagate_error(output)

            current_error = abs(float(output[0][0]) - float(self.dataset_outputs[0][0]))
            if epoch % 100 == 0:
                print('[INFO] epoch=%d, error=%.4f' % (epoch, current_error))

            if current_error < error_threshold:
                print('[INFO] Training stopped. Error is below the threshold.')
                break

    
# Test training backprop algorithm
l_rate = 0.1
n_epoch = 5000
datasetX = np.array([[2.54, 5.28, 0.78],
            [5.28, 0.78, 5.72],
            [0.78, 5.72, 0.58],
            [5.72, 0.58, 4.65],
            [0.58, 4.65, 0.91],
            [4.65, 0.91, 5.80],
            [0.91, 5.80, 1.76],
            [5.80, 1.76, 5.67],
            [1.76, 5.67, 1.73],
            [5.67, 1.73, 5.70]])
datasetY = np.array([[5.72], [0.58], [4.65], [0.91], [5.80], [1.76], [5.67], [1.73], [5.70], [1.03]])

myNetwork = NeuralNetwork(3, 4, 1, l_rate, datasetX, datasetY, n_epoch)
myNetwork.train_network()

test_dataset = np.array([[1.95, 4.18, 0.04], [4.18, 0.04, 5.05]])
test_output = np.array([[5.00], [1.79]])
prediction = myNetwork.forward_propagate(test_dataset)
for i in range(0, len(test_dataset)):
    error = np.abs(prediction[i] - test_output[i])
    print('Expected=%.2f, Got=%.4f, Error=%.4f' % (float(test_output[i][0]), float(prediction[i][0]), float(error[0])))
