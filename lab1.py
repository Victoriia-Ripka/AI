import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.wages = []
        self.alpha  = alpha 
        # реалізуємо список ваг для кожного шару
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.wages.append(w / np.sqrt(layers[i]))
            # print(self.wages)
            # print(len(layers))
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.wages.append(w / np.sqrt(layers[-2]))
            # print(len(layers))

    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
    
    def sigmoid(self, x):
		# compute and return the sigmoid activation value for a given input value
        return 1.0 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
		# compute the derivative of the sigmoid function ASSUMING
		# that x has already been passed through the 'sigmoid' function
        return np.exp(-x) / (1 + np.exp(-x)) ** 2
    
    def fit(self, X, y, epochs=1000, displayUpdate=100):
		# insert a column of 1's as the last entry in the feature
		# matrix -- this little trick allows us to treat the bias
		# as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]
		# loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
			# loop over each individual data point and train
			# our network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
			# check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
		# construct our list of output activations for each layer
		# as our data point flows through the network; the first
		# activation is a special case -- it's just the input
		# feature vector itself
        A = [np.atleast_2d(x)]
        for layer in np.arange(0, len(self.wages)):
			# feedforward the activation at the current layer by
			# taking the dot product between the activation and
			# the weight matrix -- this is called the "net input"
			# to the current layer
            net = A[layer].dot(self.wages[layer])
			# computing the "net output" is simply applying our
			# nonlinear activation function to the net input
            out = self.sigmoid(net)
			# once we have the net output, add it to our list of
			# activations
            A.append(out)
        error = A[-1] - y
        D = [error * self.sigmoid_deriv(A[-1])]
        # once you understand the chain rule it becomes super easy
		# to implement with a 'for' loop -- simply loop over the
		# layers in reverse order (ignoring the last two since we
		# already have taken them into account)
        for layer in np.arange(len(A) - 2, 0, -1):
			# the delta for the current layer is equal to the delta
			# of the *previous layer* dotted with the weight matrix
			# of the current layer, followed by multiplying the delta
			# by the derivative of the nonlinear activation function
			# for the activations of the current layer
            delta = D[-1].dot(self.wages[layer].T[:, :-1])
            delta = delta * self.sigmoid_deriv(A[layer][:, :-1])
            D.append(delta)
        D = D[::-1]
		# WEIGHT UPDATE PHASE
		# loop over the layers
        for layer in np.arange(0, len(self.wages)):
			# update our weights by taking the dot product of the layer
			# activations with their respective deltas, then multiplying
			# this value by some small learning rate and adding to our
			# weight matrix -- this is where the actual "learning" takes
			# place
            self.wages[layer] += -self.alpha * np.outer(A[layer], D[layer][:, 0])
            # швидкість начання НМ

    def predict(self, X, add_bias=True):
		# initialize the output prediction as the input features -- this
		# value will be (forward) propagated through the network to
		# obtain the final prediction
        p = np.atleast_2d(X)
		# check to see if the bias column should be added
        if add_bias:
			# insert a column of 1's as the last entry in the feature
			# matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]
		# loop over our layers in the network
        for layer in np.arange(0, len(self.wages)):
			# computing the output prediction is as simple as taking
			# the dot product between the current activation value 'p'
			# and the weight matrix associated with the current layer,
			# then passing this value through a nonlinear activation
			# function
            p = self.sigmoid(np.dot(p, self.wages[layer]))
		# return the predicted value
        return p
    
    def calculate_loss(self, X, targets):
		# make predictions for the input data points then compute
		# the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
		# return the loss
        return loss


# Define the training dataset
training_inputs = np.array([[2.54, 5.28, 0.78],
                            [5.28, 0.78, 5.72],
                            [0.78, 5.72, 0.58],
                            [5.72, 0.58, 4.65],
                            [0.58, 4.65, 0.91],
                            [4.65, 0.91, 5.80],
                            [0.91, 5.80, 1.76],
                            [5.80, 1.76, 5.67],
                            [1.76, 5.67, 1.73],
                            [5.67, 1.73, 5.70]])

training_outputs = np.array([5.72, 0.58, 4.65, 0.91, 5.80, 1.76, 5.67, 1.73, 5.70, 1.03])

# Create and train the neural network
nn = NeuralNetwork([3, 4, 1], alpha=0.5)
nn.fit(training_inputs, training_outputs, epochs=20000)

# Make predictions
for (x, target) in zip(training_inputs, training_outputs):
	# make a prediction on the data point and display the result to our console
	pred = nn.predict(x)[0][0]
	step = 1 if pred > 0.5 else 0
	print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target, pred, step))


a = nn.predict([1.73, 5.70, 1.03])
print(a)
b = nn.predict([5.70, 1.03, 5.00])
print(b)


    # @property
    # def activate(self):
    #     return self._activate
    
    # @activate.setter
    # def activate(self, sigmoid):
    #     if isinstance(sigmoid, Sigmoid):
    #         self._activate = sigmoid
    #     else:
    #         raise TypeError()
        
    # def run(self, digit1, digit2, digit3):
    #     pass



# Training the neural network
# for epoch in range(60000):
    # Forward pass (input layer to hidden layer)
    # hidden_layer_input = np.dot(training_inputs, synaptic_weights_input_hidden)
    # hidden_layer_output = sigmoid(hidden_layer_input)

    # Forward pass (hidden layer to output layer)
    # output_layer_input = np.dot(hidden_layer_output, synaptic_weights_hidden_output)
    # predicted_output = sigmoid(output_layer_input)

    # Calculate the error
    # error = training_outputs - predicted_output

    # Backpropagation
    # output_error = error * (predicted_output * (1 - predicted_output))
    # hidden_layer_error = output_error.dot(synaptic_weights_hidden_output.T) * (hidden_layer_output * (1 - hidden_layer_output))

    # Update weights
    # synaptic_weights_hidden_output += hidden_layer_output.T.dot(output_error)
    # synaptic_weights_input_hidden += training_inputs.T.dot(hidden_layer_error)

# Print the final synaptic weights
# print("Synaptic Weights (Input to Hidden):")
# print(synaptic_weights_input_hidden)

# print("\nSynaptic Weights (Hidden to Output):")
# print(synaptic_weights_hidden_output)

# Test the trained neural network with new inputs
# new_input = np.array([1.0, 2.0, 3.0])
# hidden_layer_activation = sigmoid(np.dot(new_input, synaptic_weights_input_hidden))
# output = sigmoid(np.dot(hidden_layer_activation, synaptic_weights_hidden_output))
# print("\nOutput for new input:")
# print(output)

# sigmoid_function = Sigmoid()
# neural_network = NeuralNetwork(input_number=3, neuron_number=[3, 4, 1], sigmoid=sigmoid_function)


