'''

Bryon Kucharski
Wentworth Institute of Technology
Fall 2017

Numpy implementation of a deep neural network

'''

import numpy as np
import math
import machine_learning_utils as utils

class numpy_artificial_neural_network:
    
    def __init__(self):
        self.num_layers = 0
        self.parameters = {}
        self.A_cache = {}
        self.Z_cache = {}
        self.grads = {}
    
    def sigmoid(self, z):
        return 1/(1+(np.exp(-z)))
    
    def sigmoid_derivative(z):
        return self.sigmoid(z) * (1-self.sigmoid(z))

    def relu(self,z):
        return z * (z > 0)
    
    def relu_derivative(self,z):
       return 1. * (z > 0)

    def initialize(self, dimensions):
        '''
        initalize random W martix to be (l,l-1)
        initalize b matrix to zeors size (l,1)

        Arguments:
            dimensions - size of each layer - examples [5,4,3,1]
        '''
        np.random.seed(3)

        self.num_layers = len(dimensions)

        for i in range(1,self.num_layers):
            self.parameters.update({'W' + str(i): np.random.randn(dimensions[i], dimensions[i-1]) * 0.01})
            self.parameters.update({'b' + str(i): np.zeros((dimensions[i],1))})

    
    
    def compute_forward(self, X, W, b, activation='relu'):

        print(X.shape)
        print(W.shape)
        print(b.shape)

        Z = np.add(np.dot(W,X),b)
        
        if activation == 'relu':
            A = self.relu(Z)
        elif activation == 'sigmoid':
            A = self.sigmoid(Z)

        return Z,A

    def forward_propagate(self, X, Y):
        
        A = X

        #computer rest of layers using relu
        for l in range(1, self.num_layers): #does not include the last layer (num_layers)
            
            A_prev = A
            W = self.parameters['W' + str(l)]   
            b = self.parameters['b' + str(l)]
    
            Z,A = self.compute_forward( A_prev, W, b ,'relu')

            self.A_cache.update({'A' + str(l):  A })
            self.Z_cache.update({'Z' + str(l):  Z })

        #compute last layer using sigmoid
        Z_hat, Y_hat = self.compute_forward(   A, 
                                        self.parameters['W' + str(self.num_layers-1)],
                                        self.parameters['b' + str(self.num_layers-1)],
                                        'sigmoid')

        self.A_cache.update({'A' + str(self.num_layers-1):  Y_hat })
        self.Z_cache.update({'Z' + str(self.num_layers-1):  Z_hat })

        return Y_hat

    def calculate_cost(self,Y_true , Y_hat):
        
        m = Y_true.shape[1]

        loss = (np.multiply(Y_true,np.log(Y_hat)) + (1-Y_true)*np.log(1-Y_hat))

        cost = -(loss / m )

        return cost

    def backward_propagate(self, X, Y_true, Y_hat):
        #gradiaent descent
        m = Y_true.shape[1]
        A_prev = Y_hat

        W = self.parameters['W' +  str(self.num_layers-1)]
        Z = self.parameters['Z' +  str(self.num_layers-1)]
        
        #derivative of cost function J with respect to y_hat
        dY_hat = - (np.divide(Y_true, Y_hat) - np.divide(1 - Y_true, 1 - Y_hat))

        #last layer sigmoid derivative
        dZ = dY_hat * self.sigmoid_derivative(Z)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
        dA_prev = np.dot(W.T, dZ)

        self.grads.update({'dW' + str(self.num_layers-1):  dW })
        self.grads.update({'db' + str(self.num_layers-1):  db })


        #all relu layers
        for l in reversed(range(self.num_layers-1)): #does not include the last layer (num_layers)

            W = self.parameters['W' + str(l)]
            Z = self.Z_cache['Z' +  str(l)]
            b = self.parameters['b' + str(l)]
            A_prev = self.A_cache['A' + str(l-1)]

            dZ = dA_prev * self.relu_derivative(Z)
            dW = np.dot(dZ, A_prev.T) / m
            db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
            dA_prev = np.dot(W.T, dZ)

            self.grads.update({'dW' + str(l):  dW })
            self.grads.update({'db' + str(l):  db })
            

    def update(self, learning_rate):
        for l in range(self.num_layers): 
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    def fit(self,X,Y,layers_dims, learning_rate=0.0075, num_iterations=3000):
        
        self.initialize(layers_dims)

        for i in range(0,num_iterations):

            Y_hat = self.forward_propagate(X,Y)

            cost = self.calculate_cost(Y, Y_hat)

            self.backward_propagate(X,Y, Y_hat)

            self.update(lr)
        
            if i % 100 == 0:
                        #costs.append(self.parameters['cost'])
                        #iterations.append(i)
                        print ('cost on ', i ,': ', cost)


    def predict(self):
        return ""

    def get_parameters(self):
        return self.parameters

nn = numpy_artificial_neural_network()

dims = [12288,1]

x,y = utils.load_dataset('NumpyData\Dogscats\Flattened\Subset 200\dogscats_x_train_flattened_200.npy','NumpyData\Dogscats\Flattened\Subset 200\dogscats_y_train_flattened_200.npy')
x=np.swapaxes(x,0,1)
y = np.reshape(y,(1,len(y)))

nn.fit(
        X = x,
        Y = y,
        layers_dims = dims,
        learning_rate = 0.009,
        num_iterations=3000
    )
