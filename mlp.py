import numpy as np
import pandas as pd

class MLP(object):
    def __init__(self, hidden_layers, max_iter):
        self.hidden_layers = hidden_layers
        self.n_hidden_layer = len(hidden_layers)
        self.max_iter = max_iter
        self.weights = []
        self.bias = 1

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def derivative(self, x):
        return sigmoid(x) * (1 - sigmoid(x))        

    def back_propagation(self,):
        pass

    def count_input_layer(self, training_data):
        # input layer + 1 for bias
        return len(training_data[0]) + 1
    
    def count_output_layer(self, target):
        return len(set(target))

    def init_weights(self, n_input, n_output):
        weights = []
        weights.append([[0 for col in range(self.hidden_layers[0])] for row in range(n_input)])
        for i in range(self.n_hidden_layer -1):
            weights.append([[0 for col in range(self.hidden_layers[i+1])] for row in range(self.hidden_layers[i] +1)])
        weights.append([[0 for col in range(n_output)] for row in range(self.hidden_layers[-1] +1)])
        return weights

    def create_batchs(self, training_data, batch_size):
        batchs = []
        len_data = len(training_data)
        for i in range(0, len_data, batch_size):
            if i + batch_size <= len_data:
                batchs.append(training_data[i:i+batch_size])
            else:
                batchs.append(training_data[i:])
        return batchs

    def forward(self, data):
        nodes_per_layer = []
        net_values = np.dot(data +[self.bias], self.weights[0]).tolist()
        # append out value hidden layer 1 node +bias
        nodes_per_layer.append([self.sigmoid(x) for x in net_values])
        for i in range(len(self.hidden_layers)):
            data = nodes_per_layer[i]
            net_values = np.dot(data +[self.bias], self.weights[i+1]).tolist()    
            # append out value other layer bode +bias
            nodes_per_layer.append([self.sigmoid(x) for x in net_values])
        return nodes_per_layer
    
    def error(self, output, target):
        return 1/2 * (target - output)**2

    def calc_error(self, output_layer):
        return sum([self.error(x, 1) for x in output_layer])

    def backward(self,):
        pass
        
    def fit(self,training_data, target, batch_size = 10, learning_rate = 0.001):
        # change training_data type to np.array
        training_data = training_data.values.tolist()

        n_input = self.count_input_layer(training_data)
        n_output = self.count_output_layer(target)

        # init weight 0
        self.weights = self.init_weights(n_input, n_output)

        # iteration
        for itr in range(self.max_iter):
            batchs = self.create_batchs(training_data, batch_size)
            for batch in batchs:
                # feed forward
                nodes_out = []
                # foreach data
                for i in range(len(batch)):
                    # feed forward
                    out_per_layer = self.forward(batch[i])
                    output_layer = out_per_layer[-1]
                    error_total = self.calc_error(output_layer)

                    print(error_total)
                    # Backward Phase
                    return 
                    # weight nya di rata - rata baru di update
                    