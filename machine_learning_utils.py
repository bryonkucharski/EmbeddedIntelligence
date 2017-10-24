import glob
from skimage import color, exposure, transform, io
import numpy as np
import pandas as pd
import os 
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Lambda
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')
np.set_printoptions(threshold=np.nan)
import matplotlib
from matplotlib import pyplot
import matplotlib.patches as mpatches

#https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

#from fast.ai course
vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))
calculated_mean = np.array([124.57197905557038, 116.11287913542041, 106.35763620472282]).reshape((3,1,1))
calculated_std = np.array([ 29.61407124, 28.21495712, 29.42529447]).reshape((3,1,1))

def resize_img(img, IMG_SIZE):
    '''
    resizes the images to a standard size
    '''
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant', preserve_range=True)

    return img

def roll_image(img):
     # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    return img

def preprocess_img_keras(img, IMG_SIZE):
    '''
    resizes the images to a standard size and rearranges the array dimensions
    '''

    img = resize_img(img, IMG_SIZE)
    print("-Image After resize-\n")
    print("Image Shape: ", img.shape)
    print(img)
    # roll color axis to axis 0
    img = roll_image(img)
    print("-Image After roll-\n")
    print("Image Shape: ", img.shape)
    print(img)

    img[0,:,:] = np.divide(np.subtract(img[0,:,:],calculated_mean[0]),calculated_std[0])
    img[1,:,:] = np.divide(np.subtract(img[1,:,:],calculated_mean[1]),calculated_std[1])
    img[2,:,:] = np.divide(np.subtract(img[2,:,:],calculated_mean[2]),calculated_std[2])

    print("-Image After Normalization-\nSubtratced " , calculated_mean, "\nDivided ", calculated_std)
    print("Image Shape: ", img.shape)
    print(img)
    return img

def reshape_data(data, IMG_SIZE):

        data = data.reshape(data.shape[0], IMG_SIZE*IMG_SIZE*3).astype('float32')

        return data

def preprocess_img_scikit(img, IMG_SIZE):
    
    #img = transform.resize(img,(1,IMG_SIZE*IMG_SIZE*3),mode = 'constant')
    img = resize_img(img, IMG_SIZE)
    print("-Image After resize-\n")
    print("Image Shape: ", img.shape)
    print(img)

    img[:,:,0] = np.divide(np.subtract(img[:,:,0],calculated_mean[0]),calculated_std[0])
    img[:,:,1] = np.divide(np.subtract(img[:,:,1],calculated_mean[1]),calculated_std[1])
    img[:,:,2] = np.divide(np.subtract(img[:,:,2],calculated_mean[2]),calculated_std[2])
    print("-Image After Normalization-\nSubtracted " , calculated_mean, "\nDivided ", calculated_std)
    print("Image Shape: ", img.shape)
    print(img)

    img = img.reshape(IMG_SIZE*IMG_SIZE*3).astype('float32')
    print("-Image After Reshape-\n")
    print("Image Shape: ", img.shape)
    print(img)
    return img

def get_metrics(path, extension, IMG_SIZE):
    '''
    returns mean, std of an image dataset
    '''
    mean = [0,0,0]
    M2 = [0,0,0]

    paths =  glob.glob(os.path.join(path, '*.*.' + extension)) #contains all cats and all dog images
    i = 0
    for path in paths: 
        img = io.imread(path)
        img = resize_img(img,IMG_SIZE)
        mean[0], M2[0] = calculate_metrics(img, 0, mean[0], M2[0],i+1 )
        mean[1], M2[1] = calculate_metrics(img, 1, mean[1], M2[1],i+1 )
        mean[2], M2[2] = calculate_metrics(img, 2, mean[2], M2[2],i+1 )

        if i % 100 == 0:
            print(i, 'Mean: ', mean, 'M2: ', M2)

        i = i + 1
   
    var = np.divide(M2,i-1)
    std = np.sqrt(var)

    return mean, std

def calculate_metrics(img, index, old_mean,old_M2, n):
    '''
    Welfords Algorithm to find variance and mean in one iteraton of each dimension of an image 

    img is in the format IMG_SIZE, IMG_SIZE, 3
    index - 0, 1, or 2 for R, G, B

    '''
    x = np.mean(img[:,:,index]) #returns the mean of the current image in the dimension specificed by index
    delta = x - old_mean
    new_mean = old_mean + (delta/n)
    delta2 = x - new_mean
    new_M2 = old_M2 + (delta*delta2)
    return new_mean, new_M2

def getLabel(path):
    '''
    returns 1 for cat and 0 for dog
    '''
    type = path.split('.')[0].split('/')[-1] # split string to get 'cat' or 'dog'
    print(type)
    return int(type == 'dog')

def load_image_dataset(x_train, y_train, x_valid, y_valid):
    

    print('getting image data from npy files. . .')

    x_train = np.load(x_train)
    y_train = np.load(y_train)
    x_valid = np.load(x_valid)
    y_valid = np.load(y_valid)
      
    return x_train, y_train, x_valid, y_valid

def get_image_dataset(path, extension, x_name, y_name,IMG_SIZE, modelType = 'keras'):
    '''
    gets the data and saves it as a numpy array
    '''
    
    print('getting data from directory. . .')

    x,y = parse_image_dataset(path,extension,IMG_SIZE,modelType)

    np.save(x_name, x)
    np.save(y_name, y)

    print('-Data set -\nX Shape', x.shape, '\nY Length: ' , len(y))


def parse_image_data(path, extension,IMG_SIZE, modelType = 'keras'):
    '''
    returns an numpy array of images for x and list of labels for y
    '''
    images = []
    labels = []
    paths =  glob.glob(os.path.join(path, '*.*.' + extension)) #contains all cats and all dog images
    np.random.shuffle(paths) #shuffle to prevent overfitting
    #for path in paths: 
    #    img = io.imread(path)
    for i in range(0,1):
        img = io.imread(paths[i])
        print("Original Image From Folder: ")
        print("Image Shape: ", img.shape)
        print(img)
        if modelType == 'keras':
            img = preprocess_img_keras(img, IMG_SIZE)
        elif modelType == 'scikit':
            img = preprocess_img_scikit(img, IMG_SIZE)
        images.append(img)
        label = getLabel(path)
        labels.append(label)
        print(path, label)
        
    x = np.array(images, dtype='float32')
    y = labels

    return x,y

def parse_vector_dataset(path):
    with open(path, "r") as ins:
        data = []
        labels = []
        for line in ins:

            #get rid of new line, split by comma
            list = line.replace('\n','').split(',')

            #convert string to float
            list = [float(x) for x in list]

            #add to data excluding first entry since it's the label
            data.append(list[1:])

            #add label
            labels.append(int(list[0]))

    return np.array(data), labels

def load_vector_dataset(x_name, y_name):
    
    print('getting vector data from npy files. . .')
  
    x = np.load(x_name)
    y = np.load(y_name)
    
    return x, y
    
def get_vector_dataset(path, x_name, y_name):
    
    x,y = parse_vector_dataset(path)

    np.save(x_name,x)
    np.save(y_name,y)

    print('-Data set -\nX Shape', x.shape, '\nY Length: ' , len(y))
    
def one_hot(y,NUM_CLASSES):
    '''
    turns a list into a one hot matrix
    '''
    y = np.eye(NUM_CLASSES, dtype='uint8')[y]
    return y

def reverse_one_hot(one_hot_array):
    '''
    turns a one hot matrix back into a 1D array
    '''
    return np.apply_along_axis(get_argmax, axis=1, arr = one_hot_array)

def get_argmax(input):
    return np.argmax(input)

def largeCNN_deepModel(NUM_CLASSES, IMG_SIZE):
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(3,IMG_SIZE,IMG_SIZE), activation='relu'))
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

def CNN_deepModel(NUM_CLASSES, IMG_SIZE):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(3, IMG_SIZE, IMG_SIZE), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def Keras_Website_Model(NUM_CLASSES, IMG_SIZE):

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3,IMG_SIZE,IMG_SIZE)))
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
	model.add(Dense(NUM_CLASSES))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-3), metrics=['accuracy'])
	return model

# Mean of each channel as provided by VGG researchers


def vgg_preprocess(x):
    '''
        from fast.ai course
    '''
    x = (x - vgg_mean).astype('float32')     # subtract mean
    #x = np.subtract(x,vgg_mean).astype('float32')
    #return x[:, ::-1]    # reverse axis bgr->rgb
    return x

def showImage(img):

    display = np.transpose(img, (2, 1, 0)) #rearrange to img_size,imgsize,3 so imshow can display the image
    z = np.copy(display).astype('uint8')
    pyplot.imshow(z)
    pyplot.show()

def plotPCA(pca, y):
    color = 'y'
    for i in range(len(pca)):
        if(y[i] == 1):
            color = 'b'
        elif(y[i]==2):
            color = 'r'
        elif(y[i] ==3):
            color = 'g'
        pyplot.plot(pca[i][0],pca[i][1], marker='o', ms = 5, alpha=1, color=color)
    patches = []
    patches.append(mpatches.Patch(color='b', label='1'))
    patches.append(mpatches.Patch(color='r', label='2'))
    patches.append(mpatches.Patch(color='g', label='3'))
    pyplot.legend(handles=patches, bbox_to_anchor=(1, 1), bbox_transform=pyplot.gcf().transFigure)
    pyplot.show()
