'''

Bryon Kucharski
Wentworth Institute of Technology
Fall 2017

Numpy implementation of a neural network

'''

import numpy as np
import math
import machine_learning_utils as utils


class numpy_artificial_neural_network:
    
    def __init__(self):
        self.parameters = {}

    def initialize(self):
        return ""

    def forward_propagate(self):
        
        sum = calculate_loss()

        cost = -(sum / m )
        self.parameters.update({'cost': cost, 'Y_hat': Y_hat})
        return ""

    def calculate_loss(self):
        return ""

    def backward_propagate(self):
        return ""

    def update(self):
        return ""

    def fit(self):
        return ""

    def predict(self):
        return ""

    def get_parameters(self):
        return self.parameters