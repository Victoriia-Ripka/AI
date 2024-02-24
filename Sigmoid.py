import numpy as np

class Sigmoid:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))