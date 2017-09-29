'''
Bryon Kucharski
Wentworth Institute of Technology
September 25h, 2017

Trains a model based on the following tutorial
https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

'''
from keras.datasets import mnist
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')
import numpy as np
import matplotlib
from matplotlib import pyplot

modelType = 'large_CNN'
seed = 7
np.random.seed(seed)

(x_train, y_train ), (x_test, y_test) = mnist.load_data()

#flatten
reshape_pixels = x_train.shape[1] * x_train.shape[2] #28x28

if(modelType == 'basic'):
     # reshape to be [samples][pixelx*pixels]
    x_train = x_train.reshape(x_train.shape[0], reshape_pixels).astype('float32') ## reshapes to (60000,28*28)
    x_test = x_test.reshape(x_test.shape[0],reshape_pixels).astype('float32')
elif(modelType == 'CNN' or modelType == 'large_CNN'):
    # reshape to be [samples][pixels][width][height]
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32') # 1 since greyscale images, would be 3 if using RGB
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

print('x_train',x_train.shape)
print('x_test',x_test.shape)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def basic_deepModel():
    model = Sequential()
    model.add(Dense(reshape_pixels, input_dim = reshape_pixels,activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) #between 0 and 1

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def CNN_deepModel():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def largeCNN_deepModel():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if(modelType == 'basic'):
    model = basic_deepModel()
elif(modelType == 'CNN'):
    model = CNN_deepModel()
elif(modelType == 'large_CNN'):
    model = largeCNN_deepModel()



model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))



