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
    
    def sigmoid_derivative(self,z):
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

        print('X: ' + str(X.shape))
        print('W: ' + str(W.shape))
        print('b: ' + str(b.shape))

        Z = np.add(np.dot(W,X),b)
        print('Z: ' + str(Z.shape) + '\n')
        
        if activation == 'relu':
            A = self.relu(Z)
        elif activation == 'sigmoid':
            A = self.sigmoid(Z)

        return Z,A

    def forward_propagate(self, X, Y):
        
        A = X
        L = len(self.parameters)//2

        #computer rest of layers using relu
        for l in range(1, L): #does not include the last layer (num_layers)
            print('l' +str(l))
            A_prev = A
            W = self.parameters['W' + str(l)]   
            b = self.parameters['b' + str(l)]
    
            Z,A = self.compute_forward( A_prev, W, b ,'relu')

            self.A_cache.update({'A' + str(l):  A })
            self.Z_cache.update({'Z' + str(l):  Z })

        #compute last layer using sigmoid
        Z_hat, Y_hat = self.compute_forward(   A, 
                                        self.parameters['W' + str(L)],
                                        self.parameters['b' + str(L)],
                                        'sigmoid')

        self.A_cache.update({'A' + str(L):  Y_hat })
        self.Z_cache.update({'Z' + str(L):  Z_hat })

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
        L = len(self.A_cache)

        W = self.parameters['W' +  str(L)]
        Z = self.Z_cache['Z' +  str(L)]
        
        #derivative of cost function J with respect to y_hat
        dY_hat = - (np.divide(Y_true, Y_hat) - np.divide(1 - Y_true, 1 - Y_hat))

        #last layer sigmoid derivative
        dZ = dY_hat * self.sigmoid_derivative(Z)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
        dA_prev = np.dot(W.T, dZ)

        self.grads.update({'dW' + str(L):  dW })
        self.grads.update({'db' + str(L):  db })


        #all relu layers
        for l in reversed(range(L-1)): #does not include the last layer (num_layers)
            print('back l' +str(l)+ str(L))
            W = self.parameters['W' + str(l + 1)]
            Z = self.Z_cache['Z' +  str(l + 1)]
            b = self.parameters['b' + str(l + 1)]
            A_prev = self.A_cache['A' + str(l+1)]

            dZ = dA_prev * self.relu_derivative(Z)
            dW = np.dot(dZ, A_prev.T) / m
            db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
            dA_prev = np.dot(W.T, dZ)

            self.grads.update({'dW' + str(l+1):  dW })
            self.grads.update({'db' + str(l+1):  db })
            

    def update(self, learning_rate):
        L = len(self.parameters) // 2
        for l in range(L): 
            print(self.grads["dW" + str(l+1)].shape)
            print(self.parameters["W" + str(l+1)].shape)
           # self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - learning_rate * self.grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - learning_rate * self.grads["db" + str(l+1)]

    def fit(self,X,Y,layers_dims, learning_rate=0.0075, num_iterations=3000):
        
        self.initialize(layers_dims)

        for i in range(0,num_iterations):

            Y_hat = self.forward_propagate(X,Y)

            cost = self.calculate_cost(Y, Y_hat)

            self.backward_propagate(X,Y, Y_hat)

            self.update(learning_rate)
        
            if i % 100 == 0:
                        #costs.append(self.parameters['cost'])
                        #iterations.append(i)
                        print ('cost on ', i ,': ', cost)


    def predict(self):
        return ""

    def get_parameters(self):
        return self.parameters


