import itertools, random
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
        self.unique_target = []

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def count_input_layer(self, training_data):
        # input layer + 1 for bias
        return len(training_data[0]) + 1
    
    def count_output_layer(self, target):
        return len(set(target))
    
    def error(self, output, target):
        return 1/2 * (target - output)**2

    def calc_error(self, output_layer, targets):
        return sum([self.error(output_layer[i], targets[i]) for i in range(len(output_layer))])

    def init_weights(self, n_input, n_output):
        weights = []
        random.seed(13517)
        weights.append([[random.uniform(0,1) for col in range(self.hidden_layers[0])] for row in range(n_input)])
        for i in range(self.n_hidden_layer -1):
            weights.append([[random.uniform(0,1) for col in range(self.hidden_layers[i+1])] for row in range(self.hidden_layers[i] +1)])
        weights.append([[random.uniform(0,1) for col in range(n_output)] for row in range(self.hidden_layers[-1] +1)])
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
        net_values = np.dot(data +[self.bias], self.weights[0])
        # append out value hidden layer 1 node +bias
        nodes_per_layer.append([self.sigmoid(x) for x in net_values])
        for i in range(self.n_hidden_layer):
            data = nodes_per_layer[i]
            net_values = np.dot(data +[self.bias], self.weights[i+1]).tolist()    
            # append out value other layer bode +bias
            nodes_per_layer.append([self.sigmoid(x) for x in net_values])
        return nodes_per_layer
    
    def backward(self, data, output_per_layer, weight, target, learning_rate):
        delta_err = deepcopy(output_per_layer)
        # delta output layer
        output_layer = deepcopy(output_per_layer[-1])
        delta_err[-1] = [output_layer[i]*(1-output_layer[i])*(target[i]-output_layer[i]) for i in range(len(output_layer))]
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
    
    def build_target(self, target):
        self.unique_target = list(set(target))
        targets = [0 for i in range(len(self.unique_target) -1)] + [1]
        targets = list(set(itertools.permutations(targets,len(self.unique_target))))
        dict, new_t = {}, []
        for i in range(len(self.unique_target)): dict[self.unique_target[i]] = targets[i] 
        for i in range(len(target)): 
            new_t.append(dict.get(target[i]))
        return new_t        
        
    def fit(self,training_data, target, batch_size = 10, learning_rate = 0.001, threshold=0.0001):
        # change training_data type to np.array
        training_data = training_data.values.tolist()

        n_input = self.count_input_layer(training_data)
        n_output = self.count_output_layer(target)

        targets = self.build_target(target)
        # init weight 0
        self.weights = self.init_weights(n_input, n_output)

        # iteration
        itr, error_total = 0, 99
        while itr < self.max_iter and error_total > threshold:
            i = 0 # data index
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
                    # print("awal","\n=>",temp_weights,"\n")
                    back = self.backward(mini_batch[it], hid_out_layer, self.weights, targets[i], learning_rate)
                    temp_weights = self.operation(temp_weights, op='+', arr_mat2=back)
                    error_total = self.calc_error(hid_out_layer[-1], targets[i])
                    i += 1
                # update weights
                # weight nya di rata - rata baru di update
                temp_weights = self.operation(temp_weights,val=len(batchs[b]), op='/')
                self.weights = self.operation(self.weights, op='-', arr_mat2=temp_weights)
                # self.weights = deepcopy(temp_weights)
                # print("batch", b,"\n=>",self.weights,"\n")
            itr += 1
            # print(itr)
                        
    def predict(self, data_test):
        result = []
        for i in range(len(data_test)):
            res = self.forward(data_test[i])[-1]
            result.append(self.unique_target[res.index(max(res))])
            # print(i,res)
        return result

    def score(self, data_test, label_test):
        results = self.predict(data_test)
        count_right = 0
        n_data = len(data_test)
        
        for result, label in zip(results, label_test):
            if (result == label):
                count_right = count_right + 1
        
        return count_right/n_data
