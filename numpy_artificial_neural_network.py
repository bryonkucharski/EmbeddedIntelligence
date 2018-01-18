
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
           # print('init Layer l: ' +str(i))
            
            self.parameters.update({'W' + str(i): np.random.randn(dimensions[i], dimensions[i-1]) * 0.01})
            self.parameters.update({'b' + str(i): np.zeros((dimensions[i],1))})
            
        
    def compute_forward(self, X, W, b, activation='relu'):


        Z = np.add(np.dot(W,X),b)
        #print('Z: ' + str(Z.shape) + '\n')
        
        if activation == 'relu':
            A = self.relu(Z)
        elif activation == 'sigmoid':
            A = self.sigmoid(Z)

        assert(Z.shape == (W.shape[0], X.shape[1]))

        return Z,A

    def forward_propagate(self, X, Y):
        
        A = X
        self.A_cache.update({'A0':  A })
        L = (self.num_layers-1)

        #compute rest of layers using relu
        for l in range(1, L): #does not include the last layer (num_layers)
  
            A_prev = A
            W = self.parameters['W' + str(l)]   
            b = self.parameters['b' + str(l)]

            '''
            print('forward prop input')
            print('A' + str(l-1)+ ': ' + str(A_prev.shape))
            print('W'+ str(l)+ ': ' + str(W.shape))
            print('b'+ str(l)+ ': ' + str(b.shape))
            '''

            Z,A = self.compute_forward( A_prev, W, b ,'relu')

            #print('Z'+ str(l)+ ': ' + str(Z.shape))
    
            self.A_cache.update({'A' + str(l):  A })
            self.Z_cache.update({'Z' + str(l):  Z })
        
            
        #compute last layer using sigmoid
        Z_hat, Y_hat = self.compute_forward(
                                            A, 
                                            self.parameters['W' + str(L)],
                                            self.parameters['b' + str(L)],
                                            'sigmoid'
                                            )

        #print('Z'+ str(L)+ ': ' + str(Z_hat.shape))
        self.A_cache.update({'A' + str(L):  Y_hat })
        self.Z_cache.update({'Z' + str(L):  Z_hat })

        return Y_hat
    
    def calculate_cost(self,Y_true , Y_hat):
        
        m = Y_true.shape[1]

        cost = (-1/m) * np.sum((np.multiply(Y_true,np.log(Y_hat)) + (1-Y_true)*np.log(1-Y_hat)))


        return cost

    def calculate_SGD(self,derZ, A_previous, W,m, l ):
        
        '''
        print('SGD input')
        print('derZ' + str(l) + ': ' + str(derZ.shape))
        print('A_previous' + str(l) + '.T: ' + str(A_previous.T.shape))
        print('W' + str(l) + '.T: ' + str(W.T.shape))
        '''

        dW = np.dot(derZ, A_previous.T) / m
        db = (np.sum(derZ, axis=1, keepdims=True) / m)
        dA_prev = np.dot(W.T, derZ)

        '''
        print('SGD output')
        print('dA_prev' + str(l) + ': ' + str(dA_prev.shape))
        print('dW' + str(l) +': ' + str(dW.shape))
        print('db'+ str(l) + ': ' + str(db.shape))
        '''
        
        assert (dA_prev.shape == A_previous.shape)
        assert (dW.shape == W.shape)
        
        return dA_prev, dW, db

    def backward_propagate(self, X, Y_true, Y_hat):
        
        '''
        input dA (calculated from y_hat)
        output dA_prev, dW, db for each layer
        '''
        #gradiaent descent
        m = Y_true.shape[1]
        A_prev = Y_hat
        L = (self.num_layers-1)
        Y_true = Y_true.reshape(Y_hat.shape)
        #print('backprop output Layer L: ' +str(L))
        
        W = self.parameters['W' +  str(L)]
        Z = self.Z_cache['Z' +  str(L)]
        
        #derivative of cost function J with respect to y_hat
        dY_hat = - (np.divide(Y_true, Y_hat) - np.divide(1 - Y_true, 1 - Y_hat))

        #last layer sigmoid derivative
        dZ = dY_hat * self.sigmoid_derivative(Z) # element wise multiplication 
        '''
        dW = np.dot(dZ, A_prev.T) / m
        db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
        dA_prev = np.dot(W.T, dZ)
        '''
        #print('backward prop SGD for output layer')
        A_prev = self.A_cache['A' + str(L-1)]
        dA_prev, dW, db = self.calculate_SGD(derZ = dZ, A_previous = A_prev, W = W,m = m, l = L)

        self.grads.update({'dW' + str(L):  dW })
        self.grads.update({'db' + str(L):  db })

        
        #all relu layers
        for l in reversed(range(1,L)): #does not include the last layer (num_layers)
            #print('backprop hidden Layer l: ' +str(l))
            
            W = self.parameters['W' + str(l)]
            Z = self.Z_cache['Z' +  str(l)]
            A_prev = self.A_cache['A' + str(l-1)]

            dZ = dA_prev * self.relu_derivative(Z) # element wise multiplication 
            #print('backward prop SGD for hidden layer')
            dA_prev, dW, db = self.calculate_SGD(derZ = dZ, A_previous = A_prev, W = W,m = m, l = l)

            '''
            dW = np.dot(dZ, A_prev.T) / m
            db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
            dA_prev = np.dot(W.T, dZ)
            '''
            self.grads.update({'dW' + str(l):  dW })
            self.grads.update({'db' + str(l):  db })
            

    def update(self, learning_rate):
        
        for l in range(1,self.num_layers): 

            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * self.grads["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * self.grads["db" + str(l)]
            

    def fit(self,X,Y, X_valid, Y_valid, layers_dims, learning_rate=0.0075, num_iterations=3000):
        
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

            

        train_pred = self.predict(X)
        train_accuracy = 100 - np.mean(np.abs(train_pred - Y)) * 100
        self.parameters.update({'train accuracy': train_accuracy})
        print('Train Accuracy: ' + str(train_accuracy))

        valid_pred = self.predict(X_valid)
        print(valid_pred)
        print(Y_valid)
        valid_accuracy = 100 - np.mean(np.abs(valid_pred - Y_valid)) * 100
        self.parameters.update({'valid accuracy': valid_accuracy})
        print('Validation Accuracy: ' + str(valid_accuracy))



    def predict(self, X):
        

        A = X
        L = (self.num_layers-1)

        for l in range(1, L): #does not include the last layer (num_layers)
  
            A_prev = A
            W = self.parameters['W' + str(l)]   
            b = self.parameters['b' + str(l)]

            Z,A = self.compute_forward( A_prev, W, b ,'relu')

        Z_hat, Y_hat = self.compute_forward(
                                        A, 
                                        self.parameters['W' + str(L)],
                                        self.parameters['b' + str(L)],
                                        'sigmoid'
                                        )

        pred = (Y_hat  > 0.5).astype(int)

        return pred

    def get_parameters(self):
        return self.parameters
'''
nn = numpy_artificial_neural_network()

dims = [12288,7,1]

x,y = utils.load_dataset('NumpyData\Dogscats\Flattened\Subset 200\dogscats_x_train_flattened_200.npy','NumpyData\Dogscats\Flattened\Subset 200\dogscats_y_train_flattened_200.npy')
x_valid, y_valid = utils.load_dataset('NumpyData\Dogscats\Flattened\Subset 200\dogscats_x_valid_flattened_50.npy','NumpyData\Dogscats\Flattened\Subset 200\dogscats_y_valid_flattened_50.npy')

x=np.swapaxes(x,0,1)
y = np.reshape(y,(1,len(y)))
x_valid=np.swapaxes(x_valid,0,1)
y_valid = np.reshape(y_valid,(1,len(y_valid)))

nn.fit(
        X = x,
        Y = y,
        X_valid = x_valid,
        Y_valid = y_valid,
        layers_dims = dims,
        learning_rate = 0.009,
        num_iterations= 1000
    )
'''