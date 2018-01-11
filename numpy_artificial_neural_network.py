'''

Bryon Kucharski
Wentworth Institute of Technology
Fall 2017

Numpy implementation of a deep neural network

'''

import numpy as np
import math

class numpy_artificial_neural_network:
    
    def __init__(self):
        self.num_layers = 0
        self.parameters = {}
        self.A_cache = {}
        self.grads = {}
    
    def sigmoid(self, z):
        return 1/(1+(np.exp(-z)))

    def relu(self,z):
        return np.maximum(z, 0)

    def initialize(self, dimensions):
        '''
        initalize random W martix to be (l,l-1)
        initalize b matrix to zeors size (l,1)

        Arguments:
            dimensions - size of each layer - examples [5,4,3,1]
        '''
        np.random.seed(3)

        self.num_layers = len(dimensions)

        for i in range(1,num_layers):
            self.parameters.update({'W' + str(i): np.random.randn(dimensions[i], dimensions[i-1]) * 0.01})
            self.parameters.update({'b' + str(i): np.zeros((dimensions[i],1))})
    
    
    def compute_forward(self, X, W, b, activation='relu'):

        Z = np.add(np.dot(W,X),b)

        if activation == 'relu':
            A = self.relu(A)
        elif activation == 'sigmoid':
            A = self.sigmoid(A)

        return A

    def forward_propagate(self, X, Y):
        
        A = X

        #computer rest of layers using relu
        for l in range(1, self.num_layers): #does not include the last layer (num_layers)
            
            A_prev = A
            W = self.parameters['W' + str(l)]   
            b = self.parameters['b' + str(l)]
    
            A = self.compute_forward( A_prev, W, b ,'relu')

            self.A_cache.update({'A' + str(l): , A })


        #computer last layer using sigmoid
        Y_hat = self.compute_forward(   A, 
                                        self.parameters['W' + str(self.num_layers)],
                                        self.parameters['b' + str(self.num_layers)],
                                        'sigmoid')

        self.A_cache.update({'A' + str(self.num_layers): , Y_hat })

        return Y_hat

    def calculate_cost(self,Y_true , Y_hat):
        
        m = Y_true.shape[1]

        loss = (np.multiply(Y_true,np.log(Y_hat)) + (1-Y_true)*np.log(1-Y_hat))

        cost = -(loss / m )

        return cost

    def backward_propagate(self, X, Y_true):
        
        m =   m = Y_true.shape[1]

        for l in range(1, self.num_layers): #does not include the last layer (num_layers)

            Y_hat = self.A_cache['A' + str(l)]
            dZ = Y_hat - Y_true
            dW = (1/m) * dZ * Y_hat
            dB = (1/2)*np.sum(dZ, axis = 1, keepdims = True)
            

    def update(self):
        return ""

    def fit(self,X,Y,lr, num_iterations):
        
        self.initialize()

        for i in range(0,num_iterations):

            Y_hat = self.forward_propagate(X,Y)

            cost = self.calculate_cost(Y, self.parameters['Y_hat'])

            self.backward_propagate(X,Y)

            self.update()



    def predict(self):
        return ""

    def get_parameters(self):
        return self.parameters

nn = numpy_artificial_neural_network()
