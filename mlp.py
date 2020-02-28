import numpy as np
import pandas as pd
from copy import copy, deepcopy


class MLP(object):
    def __init__(self, hidden_layers, max_iter):
        self.hidden_layers = hidden_layers
        self.n_hidden_layer = len(hidden_layers)
        self.max_iter = max_iter
        self.weights = []
        self.bias = 1

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def count_input_layer(self, training_data):
        # input layer + 1 for bias
        return len(training_data[0]) + 1
    
    def count_output_layer(self, target):
        return len(set(target))

    def init_weights(self, n_input, n_output):
        weights = []
        weights.append([[0. for col in range(self.hidden_layers[0])] for row in range(n_input)])
        for i in range(self.n_hidden_layer -1):
            weights.append([[0. for col in range(self.hidden_layers[i+1])] for row in range(self.hidden_layers[i] +1)])
        weights.append([[0. for col in range(n_output)] for row in range(self.hidden_layers[-1] +1)])
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
        for i in range(self.n_hidden_layer):
            data = nodes_per_layer[i]
            net_values = np.dot(data +[self.bias], self.weights[i+1]).tolist()    
            # append out value other layer bode +bias
            nodes_per_layer.append([self.sigmoid(x) for x in net_values])
        return nodes_per_layer
    
    def error(self, output, target):
        return 1/2 * (target - output)**2

    def calc_error(self, output_layer):
        return sum([self.error(x, 1) for x in output_layer])

    def backward(self, data, output_per_layer, weight, learning_rate):
        delta_err = deepcopy(output_per_layer)
        # delta output layer
        target = 1
        output_layer = output_per_layer[-1].copy()
        delta_err[-1] = [x*(1-x)*(target-x) for x in output_layer]
        # delta hidden layer
        for i in range(self.n_hidden_layer -1, -1, -1):
            for j in range(len(delta_err[i])):
                sigma_w_delta = np.dot(weight[i+1][j], delta_err[i+1])
                out = output_per_layer[i][j]
                delta_err[i][j] = out * (1 - out) * sigma_w_delta
                
        delta_w = deepcopy(weight)
        # add bias
        new_weight = deepcopy(weight)
        data.append(self.bias)
        flag = 0
        for k in range(len(weight)):
            if flag != 0:
                data = output_per_layer[k-1]
                data.append(self.bias)
            for row in range(len(weight[k])):
                for col in range(len(weight[k][row])):
                    delta_w[k][row][col] = learning_rate * delta_err[k][col] * data[row]
                    new_weight[k][row][col] = weight[k][row][col] - delta_w[k][row][col]
            flag += 1

        return new_weight

    def operation(self, arr_mat1, op, arr_mat2 = None, val = None):
        for i in range(len(arr_mat1)):
            for j in range(len(arr_mat1[i])):
                for k in range(len(arr_mat1[i][j])):
                    if op == '+' and val == None:
                        arr_mat1[i][j][k] += arr_mat2[i][j][k]
                    if op == '-' and val == None:
                        arr_mat1[i][j][k] -= arr_mat2[i][j][k]
                    if op == '/' and val != None:
                        arr_mat1[i][j][k] /= val
        return arr_mat1
        
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
            for b in range(len(batchs)):
                mini_batch = deepcopy(batchs[b])
                # temporary weights untuk perhitungan weight rata-rata
                temp_weights = deepcopy(self.weights)
                # foreach data
                for it in range(len(batchs[b])):
                    # feed forward
                    hid_out_layer = self.forward(mini_batch[it])
                    # Backward Phase
                    temp_weights = self.operation(temp_weights, op='+', arr_mat2=self.backward(mini_batch[it], hid_out_layer, self.weights, learning_rate))
                # update weights
                # weight nya di rata - rata baru di update
                temp_weights = self.operation(temp_weights,val=len(batchs[b]), op='/')
                self.weights = self.operation(self.weights, op='-', arr_mat2=temp_weights)
                print(itr,"\n",self.weights,"\n")
                    