import glob
from skimage import color, exposure, transform, io
import numpy as np
import pandas as pd
import os 
import re
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
K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_first')
np.set_printoptions(threshold=np.nan)
import matplotlib
from matplotlib import pyplot
import matplotlib.patches as mpatches
import random
from sklearn.metrics import confusion_matrix
import itertools

#https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

#from fast.ai course
vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))
calculated_mean = np.array([124.57197905557038, 116.11287913542041, 106.35763620472282]).reshape((3,1,1))
calculated_std = np.array([ 29.61407124, 28.21495712, 29.42529447]).reshape((3,1,1))

def resize_img(img, IMG_SIZE):
    """
    Resizes image to IMG_SIZE, IMG_SIZE

    Args:
        img: input image 
        IMG_SIZE: size to reshape the image 
    Returns:
        resized image
    """
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant', preserve_range=True)

    return img

def roll_image(img):
    """
    Rolls an axis back negative one
    used to turn an image such as (3,224,224) to (224,224,3)

    Args:
        img: input image 
    Returns:
        rolled image
    """
    img = np.rollaxis(img, -1)
    return img

def preprocess_img_keras(img, IMG_SIZE, standardize = True):
    """
    Resizes image to IMG_SIZE, IMG_SIZE
    Rolls image
    standardizes images in R,B,G channels based on output of get_metrics

    Args:
        img: input image format (3,size,size)
        IMG_SIZE: size to reshape the image 
        standardize: boolean to standardize or not
    Returns:
        preprocessed image size (IMG_SIZE,IMG_SIZE,3)
    """

    img = resize_img(img, IMG_SIZE)

    # roll color axis to axis 0
    img = roll_image(img)
    if standardize:
        img[0,:,:] = np.divide(np.subtract(img[0,:,:],calculated_mean[0]),calculated_std[0])
        img[1,:,:] = np.divide(np.subtract(img[1,:,:],calculated_mean[1]),calculated_std[1])
        img[2,:,:] = np.divide(np.subtract(img[2,:,:],calculated_mean[2]),calculated_std[2])

    return img
'''
def reshape_data(data, IMG_SIZE):

        data = data.reshape(data.shape[0], IMG_SIZE*IMG_SIZE*3).astype('float32')

        return data
'''
def preprocess_img_scikit(img, IMG_SIZE,standardize = True):
    """
    *Also known as flattening the image. the Scikit library requires all flattened images
    Resizes image to IMG_SIZE, IMG_SIZE
    Rolls image
    standardizes images in R,B,G channels based on output of get_metrics

    Args:
        img: input image format (size,size,3)
        IMG_SIZE: size to reshape the image 
        standardize: boolean to standardize or not
    Returns:
        preprocessed image size (1,IMG_SIZE*IMG_SIZE*3)
    """
    
    #img = transform.resize(img,(1,IMG_SIZE*IMG_SIZE*3),mode = 'constant')
    img = resize_img(img, IMG_SIZE)
    
    if standardize:
        img[:,:,0] = np.divide(np.subtract(img[:,:,0],calculated_mean[0]),calculated_std[0])
        img[:,:,1] = np.divide(np.subtract(img[:,:,1],calculated_mean[1]),calculated_std[1])
        img[:,:,2] = np.divide(np.subtract(img[:,:,2],calculated_mean[2]),calculated_std[2])
    
    img = img.reshape(IMG_SIZE*IMG_SIZE*3).astype('float32')

    return img

def get_metrics(path, extension, IMG_SIZE):
    """
    calculates mean and std of entire dataset

    Args:
        path: path to folder of images
        IMG_SIZE: size to reshape the image 
        extension: 'jpg' , 'png' , etc.
    Returns:
        mean, std
    """
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
    """
    Welfords Algorithm to find variance and mean in one iteraton of each dimension of an image

    Args:
        img: image size IMG_SIZE, IMG_SIZE, 3
        index: 0, 1, or 2 for R, G, B
        old_mean: mean from last iteration (or 0 to start) 
        old_M2:  M2 from last iteration (or 0 to start) 
        n: number of images so far
    Returns:
        new mean and M2
    """
    x = np.mean(img[:,:,index]) #returns the mean of the current image in the dimension specificed by index
    delta = x - old_mean
    new_mean = old_mean + (delta/n)
    delta2 = x - new_mean
    new_M2 = old_M2 + (delta*delta2)
    return new_mean, new_M2

def getLabel(path):
    """
    cat and dog dataset the label was in the image name. this parses the name of the file to get either cat (0) or dog(1)
    returns 1 for cat and 0 for dog - not the best solution - needs to be tweaked if using linux machine vs windows machine

    Args:
        path: image name
    Returns:
        label
    """
    
    type = path.split('.')[0].split('/')[-1] # split string to get 'cat' or 'dog'
    print(type)
    return int(type == 'dog')

def load_dataset(x_name, y_name):
    """
    loads a .npy file x_name and y_name

    Args:
        x_name: .npy file name for x
        y_name: .npy file name for y
       
    Returns:
        x,y matricies of data
    """
    
    print('getting vector data from npy files. . .')
  
    x = np.load(x_name)
    y = np.load(y_name)

    print('-Data set -\nX Shape', x.shape, '\nY Length: ' , len(y))

    return x, y

def parse_image_data(path, extension, x_name, y_name,IMG_SIZE, modelType = 'keras', standardize = True):
    """
    this was written for the cats/dogs data set. Another parse method will probably need to be written for other data sets
    read, resize, standardize image
    get the label
    save the arrays as an npy file
    returns an numpy array of images for x and list of labels for y

    Args:
        path: path to folder of images
        IMG_SIZE: size to reshape the image 
        extension: 'jpg' , 'png' , etc.
        x_name: name to save x data once parsed
        y_name: name to save y data once parsed
        modelType: 'keras' to call preprocess_img_keras , 'scikit' to call preprocess_img_scikit
        standardize: boolean to standardize or not

    Returns:
        x,y matricies of data
    """


    print('getting data from directory. . .')

    images = []
    labels = []
    paths =  glob.glob(os.path.join(path, '*.*.' + extension)) #contains all cats and all dog images
    np.random.shuffle(paths) #shuffle to prevent overfitting
    for path in paths: 
        img = io.imread(path)
        if modelType == 'keras':
            img = preprocess_img_keras(img, IMG_SIZE,standardize)
        elif modelType == 'scikit':
            img = preprocess_img_scikit(img, IMG_SIZE, standardize)
        images.append(img)
        label = getLabel(path)
        labels.append(label)
        print(path, label)
        
    x = np.array(images, dtype='float32')
    y = labels

    np.save(x_name, x)
    np.save(y_name, y)

    print('-Data set -\nX Shape', x.shape, '\nY Length: ' , len(y))
    print('mean: ', np.mean(x),'std: ', np.std(x))
    return x,y

def parse_vector_dataset(path,x_name,y_name,position_of_label = 'first', delim = ','):
    '''
    this function often changed based on the dataset - was not able to write one function to parse every dataset
    '''
    with open(path, "r") as ins:
        data = []
        labels = []
        for line in ins:
            
            #get rid of new line, split by delim
            list = line.strip().replace('\n','')
            list = re.split(delim,list)
            
            if '?' in list:
                continue
            #convert to float
            list = [float(x) for x in list]

            if(position_of_label == 'first'):
                #add to data excluding first entry since it's the label
                data.append(list[1:])

                #add label
                labels.append(int(list[0]))

            elif(position_of_label == 'last'):
                
                data.append(list[:-1])

                #add label
                if int(list[-1]) == 2:
                    labels.append(0)
                elif int(list[-1]) == 4:
                    labels.append(1)
                else:
                    print('SOMETHING WENT WRONG')
                #labels.append(int(list[-1]))

    x = np.array(data)
    y = labels

    if(min(y) is not 0):
        #make sure labels start at 0
        y = [x - 1 for x in y]
        print('Making labels start at 0')
 
    np.save(x_name,x)
    np.save(y_name,y)

    print('-Data set -\nX Shape', x.shape, '\nY Length: ' , len(y))

    return x, y
    
def one_hot(y,NUM_CLASSES):
    """
    turns a list into a one hot matrix

    Args:
        y: y vector of labels
        NUM_CLASSES: number of classes in dataset
        
    Returns:
        one hot array of input data
    """

    y = np.eye(NUM_CLASSES, dtype='uint8')[y]
    return y

def reverse_one_hot(one_hot_array):
    """
    turns a one hot matrix back into a vector

    Args:
        one_hot_array: one hot array of y data
        
    Returns:
        y vector of labels
    """
    return np.apply_along_axis(get_argmax, axis=1, arr = one_hot_array)

'''
def get_argmax(input):
    return np.argmax(input)
'''

def largeCNN_deepModel(NUM_CLASSES, IMG_SIZE):
    """
    Larger CNN

    Args:
        NUM_CLASSES: number of classes for the output layer
        IMG_SIZE: (3,IMG_SIZE, IMG_SIZE) of input data
        
    Returns:
        keras model 
    """
    
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
    """
    Smaller CNN

    Args:
        NUM_CLASSES: number of classes for the output layer
        IMG_SIZE: (3,IMG_SIZE, IMG_SIZE) of input data
        
    Returns:
        keras model 
    """
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

def custom_Deep_Model(input_size, num_layers, num_hidden_units,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate):
    
    """
    Custom Deep Neural Network
    Architecture will be 
    input layer size input_shape  -> 1 to num_layers hidden layers size num_hidden_units (all same size) -> output size num_outputs

    Args:
        inputs to the model
        
    Returns:
        keras model 
    """
    
    model = Sequential()
   
    if optimizer == 'adam':
        opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)
        #can add other Keras optimizers here

    model.add(Dense(num_hidden_units,input_shape = input_size, activation = hidden_activation))

    for i in range(num_layers):
        model.add(Dense(num_hidden_units,activation = hidden_activation ))

    model.add(Dense(num_outputs, activation = output_activation))

    model.compile(loss=loss,optimizer=opt, metrics=['accuracy'])
    
    return model

def custom_CNN_Model(input_size, num_layers,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate, filter_size, kernal_size, pooling_size):
    
    """
    Custom Convolutional Neural Network
    Architecture will be 
    input layer size input_shape  -> 1 to num_layers hidden layers size num_hidden_units (all same size) -> output size num_outputs

    Args:
        inputs to the model
        
    Returns:
        keras model 
    """

    model = Sequential()
   
    if optimizer == 'adam':
        opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)

    model.add(Conv2D(filter_size, kernal_size, input_shape=input_size, activation=hidden_activation))
    model.add(MaxPooling2D(pool_size=pooling_size))

    for i in range(num_layers):
        model.add(Conv2D(filter_size, kernal_size))
        model.add(Activation(hidden_activation))
        model.add(MaxPooling2D(pool_size=pooling_size))

    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(num_outputs, activation = output_activation))

    model.compile(loss=loss,optimizer=opt, metrics=['accuracy'])
    
    return model

def showImage(img, title= '' ,transpose = False):
    """
    displays an image using matplotlib
    must be in shape (size,size,3)
    if image size is  (3,size,size), set transpose = true

    Args:
        img: image to show
        title: title of plot
        transpose: boolean to change image axis
    """

    if transpose:
        display = np.transpose(img, (2, 1, 0)) #rearrange to img_size,imgsize,3 so imshow can display the image
        z = np.copy(display).astype('uint8')
    else:
        z = img
    pyplot.title(title)
    pyplot.imshow(z)
    pyplot.show()

def plotPCA(x, y, dimensions,labels,xlabel = '', ylabel = '', title = ''):
    """
    Plots x vs y if dimensions is 2
    Plots x vs 0 if dimensions is 1
    plots x vs index if dimenion is 0 (used for plotting a 1D array in 2 dimensions, we did this for the breat cancer dataset for the poster)

    Args:
        x: x axis data
        y: y axis data
        dimensions: dimensions to plot in 0, 1 or 2
         
    """

    num_classes = int(max(y)) + 1
    print('Plotting ',num_classes, ' classes in ', dimensions, ' dimensions')

    clrs = ['y','b','r', 'g', 'c','m','k','w']
    for i in range(len(x)):
        color = clrs[int(y[i])]
        
        if(dimensions == 1):
            pyplot.plot(x[i],0,marker='+', ms = 1, alpha=1, color=color)
        elif(dimensions == 2):
            pyplot.plot(x[i][0],x[i][1], marker='o', ms = 5, alpha=1, color=color)
        elif(dimensions == 0): #used if wanting to plot 1D pca/lda in 2D
            pyplot.plot(x[i][0],i, marker='o', ms = 5, alpha=1, color=color)

    patches = []
    
    for i in range(0,num_classes):
        patches.append(mpatches.Patch(color=clrs[i], label=labels[i]))
 
    
    pyplot.legend(handles=patches, bbox_to_anchor=(1, 1), bbox_transform=pyplot.gcf().transFigure)
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.show()

def create_validation_set(x_train, y_train, num_validations):
    """
    Takes a training set and creates a validation set by randomly selecting indexes
    validation data points are removed from the training set

    Args:
        x_train: x_train data
        y_train: y_train data
        num_validations: how big to make the validation set - typicaly (number_train_data * .2)
         
    """
    
    x_valid = []
    y_valid = []

    #gets unique random numbers the size of num_validations
    indexes = random.sample(range(0,x_train.shape[0]-1), round(num_validations))


    for index in indexes:
        x_valid.append(x_train[index])
        y_valid.append(y_train[index])

        
    new_x_train = np.delete(x_train,indexes,0)
    new_y_train = np.delete(y_train,indexes,0)
    x_valid = np.array(x_valid, dtype='float32')

    print('-New Training Data set -\nX Shape', new_x_train.shape, '\nY Length: ' , len(new_y_train))
    print('-Validation Data set -\nX Shape', x_valid.shape, '\nY Length: ' , len(y_valid))
    
    return new_x_train,new_y_train,x_valid,y_valid


def get_confusion_matrix(y_true, y_pred, labels, title = ''):
    '''
    see below
    '''
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    pyplot.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels, title=title)
    pyplot.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Taken directly from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks = np.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45)
    pyplot.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')


def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    CODE TAKEN DIRECTLY FROM https://gist.github.com/craffel/2d727968c3aaebd10359

    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = pyplot.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = pyplot.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

def draw_nn(size, save_name):
    '''
    see above
    '''
    fig = pyplot.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, .1, .9, .1, .9, size)
    fig.savefig(save_name)

