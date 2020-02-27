import numpy as np

class MLP(object):
    def __init__(self, hidden_layer, max_iter):
        self.hidden_layer_size = hidden_layer_size
        self.epoch = max_iter
        self.weights = self.init_weights()

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def init_weights(self):
        pass

    def back_propagation(self,):
        pass


    
        