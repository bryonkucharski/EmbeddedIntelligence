'''

Bryon Kucharski
Wentworth Institute of Technology
Fall 2017

Numpy implementation of logistic regression

'''

import numpy as np
import math
import machine_learning_utils as utils


class numpy_logistic_regression():

    def __init__(self):
        self.parameters = {}
    
    def sigmoid(self, z):
        return 1/(1+(np.exp(-z)))

    def initialize(self, size):

        """
        sets W to a numpy array to zeros and sets b to 0

        Arguments:
            size - size to make W so the shape will be (size, 1)
        """

        W = np.zeros(shape=(size, 1))
        b = 0

        self.parameters.update({'W': W, 'b': b})
        
    def forward(self, X,Y):
        """
        calculates the cost of a single iteration

        Arguments:
            X - input of dimensions (features,size)
            Y - 1D numpy array of true output values

        """
        
        W = self.parameters['W']
        b = self.parameters['b']
    
        m = X.shape[1]
        Z = np.add(np.dot(W.T,X),b)
        Y_hat = self.sigmoid(Z)
   
        sum = np.sum(self.calculate_loss(Y_hat,Y))
        
        cost = - (sum / m)

        self.parameters.update({'cost': cost, 'Y_hat': Y_hat})


    
    def calculate_loss(self, Y_hat, Y_true):
        """
        calculates the loss of a single iteration

        Arguments:
            Y_hat - 1D numpy array of sigmoid outputs
            Y_true - 1D numpy array of true output values
        """
        return (np.multiply(Y_true,np.log(Y_hat)) + (1-Y_true)*np.log(1-Y_hat))

    def backward(self,X, Y_true):
        """
        calculates the derivative terms of the cost with respect to each variable

        Arguments:
            X - input of dimensions (features,size)
            Y_true - 1D numpy array of true output values
        """
        
        W = self.parameters['W']
        b = self.parameters['b']
        
        Y_hat = self.parameters['Y_hat']

        m = len(X)

        dZ = Y_hat - Y_true
        db = (1/m)*np.sum(dZ)
        dW = (1/m)*np.dot(X,dZ.T)

        self.parameters.update({'dZ': dZ, 'db': db,'dW':dW })



    def update(self, learning_rate):
        """
        Updates W and b based using graduent descent

        Arguments:
           learning_rate - how fast gradient descent will "learn"

        """

        W = self.parameters['W']
        b = self.parameters['b']
        dW = self.parameters['dW']
        db = self.parameters['db']

        W = W - learning_rate*dW
        b = b - learning_rate*db

        self.parameters.update({'W': W, 'b': b})

    def fit(self, x, y, x_test,y_test, num_iterations, learning_rate = .05):
        """
        Run logistic regression for num_iterations. Calcuates the train and test accuracy

        Arguments:
           x - train input of dimension (features,size)
           y - 1D numpy array of true output values
           x_test - test input of dimension (feature, size)
           y_test - 1D numpy array of true test values
           num_iterations - number of times to run logistic regression
           learning_rate - how fast gradient descent will "learn"

        Returns:
            costs,iterations, test_pred - list of costs after each iteration, iteration associated with the cost, predicitons of test dataset
        

        """
        costs = []
        iterations = []
        self.initialize(x.shape[0])

        for i in range(0,num_iterations):
            
            #print(self.parameters['dW'], self.parameters['db'])

            self.forward(x, y)

            self.backward(x,y)

            self.update(learning_rate)

            if i % 100 == 0:
                    costs.append(self.parameters['cost'])
                    iterations.append(i)
                    print ('cost on ', i ,': ', self.parameters['cost'])

        train_pred = self.predict(x)
        #print('Prediciton: ', train_pred)
        #print('True: ', y)
        train_accuracy = 100 - np.mean(np.abs(train_pred - y)) * 100
        self.parameters.update({'train accuracy': train_accuracy})

        test_pred = self.predict(x_test)
        #print('test Prediciton: ', test_pred)
        #print('True: ', y_test)
        test_accuracy = 100 - np.mean(np.abs(test_pred - y_test)) * 100
        self.parameters.update({'test accuracy': test_accuracy})

        return costs, iterations, test_pred
    
   
    def predict(self,input):
        """
        binary classification of the given inputs based on calcuated W and b
        Arguments:
          input - numpy array to predict of dimension (feature, size)
        """
        W = self.parameters['W']
        b = self.parameters['b']
        Z = np.add(np.dot(W.T,input),b)
        pred = self.sigmoid(Z)
        
        results = (pred  > 0.5).astype(int)
        '''
        for i in range(pred.shape[1]):
          
            if pred[0][i] > .66:
                pred[0][i] = 2

            elif pred[0][i] < .66 and pred[0][i] > .33:
               pred[0][i] =  1
               
            elif pred[0][i] < .33:
                pred[0][i] =  0
        results = pred
        '''


        return results
        
    def get_parameters(self):
        return self.parameters
       

