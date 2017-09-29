'''

Bryon Kucharski
Wentworth Institute of Technology
September 26th, 2017


Train a model to predict a cat or dog using one of three Deep Neural Networks
Predicts using the model weights

data can be loaded from x_train_raw.npy, y_train_raw.npy, y_valid_raw.npy, and y_valid_raw.npy

If getting data from directory, Cats and dogs data must be organized dogscats/train and dogscats/valid. The get_data function will shuffle and create labels of the data
'''
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.optimizers import SGD, RMSprop, Adam
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
K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')
import numpy as np
import os
import random
import glob
import pickle
from skimage import color, exposure, transform, io
import matplotlib
from matplotlib import pyplot
import sys

NUM_CLASSES = 2
IMG_SIZE = 224
get_data_from_directory = False


def preprocess_img(img):
    '''
    resizes the images to a standard size and rearranges the array dimensions
    '''

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img

def getLabel(path):
    '''
    returns 1 for cat and 0 for dog
    '''
    type = path.split('.')[0].split('/')[2] # split string to get 'cat' or 'dog'
#    print(type)
    return int(type == 'dog')


def get_data(path, extension):
    images = []
    labels = []
    paths =  glob.glob(os.path.join(path, '*.*.' + extension)) #contains all cats and all dog images
    np.random.shuffle(paths) #shuffle to prevent overfitting
    for path in paths:    
        img = io.imread(path)
        img = preprocess_img(img)
        images.append(img)
        label = getLabel(path)
        labels.append(label)
        print(path, label)

    
    x = np.array(images, dtype='float32')
    # Make one hot targets
    y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
    #y = labels
    return x,y

def largeCNN_deepModel():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(3,224,224), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-6), metrics=['accuracy'])
    return model

def CNN():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(3, 224, 224),activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-3), metrics=['accuracy'])

    return model


def Keras_Website_Model():

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3,224,224)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-3), metrics=['accuracy'])
	return model

def train():
    if (get_data_from_directory):
        print('getting data from directory. . .')
        x_train,y_train = get_data('dogscats/train/','jpg')
        x_valid,y_valid = get_data('dogscats/valid/','jpg')

    #    x_train = x_train / 255
    #   x_valid = x_valid / 255

    #   print('Training set', x_train.shape, y_train.shape)
        #print('Validation set', x_valid.shape, y_valid.shape)
        np.save('x_train_raw', x_train)
        np.save('x_valid_raw',x_valid)
        np.save('y_train_raw',y_train)
        np.save('y_valid_raw',y_valid)


    else:
        print('getting data from npy files. . .')

        x_train = np.load('x_train_raw.npy')
        y_train = np.load('y_train_raw.npy')
        x_valid = np.load('x_valid_raw.npy')
        y_valid = np.load('y_valid_raw.npy')
        print('Training set', x_train.shape, y_train.shape)
        print('Validation set', x_valid.shape, y_valid.shape)

    model = Keras_Website_Model()

    model.fit(x_train, y_train, validation_data = (x_valid, y_valid), epochs=10, batch_size=200, verbose=2)
    scores = model.evaluate(x_valid, y_valid, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    model.save_weights("dogscats_94.h5")

def predict(path):
    image = [] #save image to a list because the model expects 4 dims in the order 1,3,img_size,img_size
    model = Keras_Website_Model()
    model.load_weights('dogscats_94.h5')

    img =io.imread(path)
    img = preprocess_img(img)
    image.append(img)
    img = np.array(image)


    pred = model.predict(img)
    print('Cat: ' , pred[0][0] , '\nDog: ' , pred[0][1])

    display = np.transpose(img[0], (2, 1, 0)) #rearrange to img_size,imgsize,3 so imshow can display the image
    print(display.shape)
    pyplot.imshow(display)
    pyplot.show()


predict('dogscats_v2/test1/661.jpg')
