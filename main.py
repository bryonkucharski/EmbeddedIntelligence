'''

Bryon Kucharski
Wentworth Institute of Technology
Fall 2017

'''

from machine_learning_classifiers import machine_learning_classifiers
import machine_learning_utils as utils
import numpy as np
from skimage import io
import random
from matplotlib import pyplot
import re

classifiers = machine_learning_classifiers()

def runKeras(parseData=False):
    
    #to run keras model
    if parseData:
        utils.parse_dataset('dogscats/train', 'jpg', 'dogscats_x_train_preprocessed', 'dogscats_y_train_preprocessed',224,'keras')
        utils.parse_dataset('dogscats/valid','jpg', 'dogscats_x_valid_preprocessed', 'dogscats_y_valid_preprocessed',224, 'keras')

    classifiers.load_dataset(r'dogscats_x_train_preprocessed.npy', r'dogscats_y_train_preprocessed.npy', r'dogscats_x_valid_preprocessed.npy', r'dogscats_y_valid_preprocessed.npy')
    classifiers.KerasDeepModel(30,'dogscats_deepmodel_preprocessed.h5')


def runSciKit(parseData = False):

    if parseData:
        utils.parse_dataset('dogscats/train', 'jpg', 'x_scikit_preprocessed', 'y_scikit_preprocessed',50,'scikit')
        utils.parse_dataset('dogscats/valid', 'jpg', 'x_test_scikit_preprocessed', 'y_test_scikit_preprocessed',50,'scikit')
    classifiers.load_dataset('x_scikit_preprocessed.npy', 'y_scikit_preprocessed.npy', 'x_test_scikit_preprocessed.npy', 'y_test_scikit_preprocessed.npy')

    print('Running LinearSVM')
    classifiers.LinearSVM()

    print('Running KNN')
    classifiers.KNN()

    print('Running Random Forest')
    classifiers.RandomForestClassifier()

    print('Running GaussianNB')
    classifiers.GaussianNB()
    '''
    print('Running Logistic Regression')
    classifiers.LogisticRegression()
    '''

def runPCA(path, dataset_x, dataset_y,parseData = False,standardize = True, type = 'scikit', n_components = 2, title = '', labels = []):
    if parseData:
        utils.parse_vector_dataset(path,dataset_x, dataset_y)
    classifiers.load_vector_dataset(dataset_x, dataset_y)
    #classifiers.numpy_PCA()
    if type == 'scikit':
        results = classifiers.scikit_PCA(n_components, labels, title)
    elif type == 'numpy':
        results = classifiers.numpy_PCA(n_components,labels, plot_title = title)
    print(results)

def runLDA(path = '', dataset_x= '', dataset_y= '', parseData = False,standardize = True, type = 'scikit', num_classes=3, num_features=13,input_name = 'None', predict_path='None', title='', labels = [] ):
    if parseData:
        utils.parse_vector_dataset(path,dataset_x, dataset_y)

    classifiers.load_vector_dataset(dataset_x, dataset_y)

    if type == 'scikit':
        results = classifiers.scikit_LDA(num_classes,labels,True)
    elif type == 'numpy':
        results = classifiers.numpy_LDA(num_classes, num_features, labels,standardize, title)
    print(results)

def runNumpyLogisticRegression(x,y,x_valid,y_valid, num_iterations = 2000, learning_rate = .05, title = '', labels=''):
    
    classifiers.load_dataset(x,y,x_valid,y_valid)
                      
    classifiers.numpy_logistic_reg(num_iterations, learning_rate, confusion_title=title, confusion_labels=labels)

def HyperparameterTuneDeepNN(iterations):
    results = []

    for i in range(iterations):
        #r = -0.2*np.random.rand()

        #for pima indians
        alpha = .02
        num_hidden_units = 70
        num_layers =10
        batch_size =200
        epochs = 200

        classifiers = machine_learning_classifiers()
    
        classifiers.load_dataset(
                            r'NumpyData\Pima Indians\pima_indians_x_train.npy',
                            r'NumpyData\Pima Indians\pima_indians_y_train.npy',
                            r'NumpyData\Pima Indians\pima_indians_x_valid.npy',
                            r'NumpyData\Pima Indians\pima_indians_y_valid.npy')
        
        result = classifiers.CustomDeepModel(input_size = (8,),
                                num_layers = num_layers, 
                                num_hidden_units = num_hidden_units,
                                num_outputs = 2,
                                output_activation = 'softmax',
                                hidden_activation = 'relu', 
                                loss = 'categorical_crossentropy',
                                optimizer = 'adam',
                                learning_rate = alpha,
                                epochs = epochs, 
                                batch_size = batch_size)
        results.append([num_layers,num_hidden_units, alpha,epochs, batch_size, result[1]])
        #pyplot.plot(num_hidden_units,result[1],marker='+', ms = 1, alpha=1, color='b')
    
    for test in results:
        print("\tnum_layers: " , test[0], "\t\tnum_hidden_units: " , test[1],"\tlearning_rate: " , test[2],"\tepochs: " , test[3],"\tbatch_size: " , test[4],  "\tAccuracy: " , test[5])
        
    #pyplot.show()
def HyperparameterTuneCNN(iterations):
    results = []

    for i in range(iterations):
        #r = -0.2*np.random.rand()

        #for pima indians
        alpha = .02
        #num_hidden_units = 70
        num_layers =1
        batch_size =200
        epochs = 200


        classifiers.load_dataset(   r'dogscats_x_train_preprocessed.npy', 
                                    r'dogscats_y_train_preprocessed.npy',
                                    r'dogscats_x_valid_preprocessed.npy', 
                                    r'dogscats_y_valid_preprocessed.npy')
                        
        result = classifiers.CustomCNNModel(
                                            input_size = (3, 224, 224),
                                            num_layers = num_layers,
                                            num_outputs = 2,
                                            output_activation = 'softmax',
                                            hidden_activation = 'relu', 
                                            loss = categorical_crossentropy ,
                                            optimizer = 'adam',
                                            learning_rate = alpha, 
                                            epochs = epochs,
                                            batch_size = batch_size,
                                            filter_size = 32, 
                                            kernal_size = (3,3), 
                                            pooling_size = (2,2)
                                            )
        #results.append([num_layers,num_hidden_units, alpha,epochs, batch_size, result[1]])
        #pyplot.plot(num_hidden_units,result[1],marker='+', ms = 1, alpha=1, color='b')
    
   # for test in results:
      #  print("\tnum_layers: " , test[0], "\t\tnum_hidden_units: " , test[1],"\tlearning_rate: " , test[2],"\tepochs: " , test[3],"\tbatch_size: " , test[4],  "\tAccuracy: " , test[5])
        
def get_data():
    with open('wine.data.csv.txt', "r") as ins:
        li = ins.readlines()

    random.shuffle(li)
    data = []
    labels = []
    for line in li:
        list = line.strip().replace('\n','')
        list = re.split(',',list)
        label = list[0]

        if label == '1':
            labels.append(0)
        elif label == '2':
            labels.append(1)
        elif label == '3':
            labels.append(2)

        info = list[1:]
        info = [float(x) for x in info]

        data.append(info)

    x = np.array(data)
    y = labels

    print('-Data set -\nX Shape', x.shape, '\nY Length: ' , len(y))

    x,y,x_valid,y_valid = utils.create_validation_set(x,y,x.shape[0]*.2)
    #x_new = x
    #x_valid_new = x_valid
    x_new = (x - np.mean(x)) / np.std(x)
    x_valid_new = (x_valid - np.mean(x)) / np.std(x)

    print('mean', np.mean(x_new))
    print('std', np.std(x_new))

    np.save('wine_x_train',x_new)
    np.save('wine_y_train',y)
    np.save('wine_x_valid',x_valid_new)
    np.save('wine_y_valid',y_valid)

def get_image_data():
    #utils.parse_image_data(path = 'dogscats/train', extension = 'jpg', x_name = 'dogscats_x_train_flattened', y_name = 'dogscats_y_train_flattened',IMG_SIZE = 64, modelType = 'scikit')
    #utils.parse_image_data(path = 'dogscats/valid', extension = 'jpg', x_name = 'dogscats_x_valid_flattened', y_name = 'dogscats_y_valid_flattened',IMG_SIZE = 64, modelType = 'scikit')
    x,y = utils.load_dataset(r'dogscats_x_train_flattened.npy',
                             r'dogscats_y_train_flattened.npy')

    x_v,y_v = utils.load_dataset(r'dogscats_x_valid_flattened.npy',
                             r'dogscats_y_valid_flattened.npy')
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    indexes = random.sample(range(0,x.shape[0]-1), 200)
    indexes2 = random.sample(range(0,x_v.shape[0]-1), 50)

    for index in indexes:
        x_train.append(x[index])
        y_train.append(y[index])

    for index in indexes2:
        x_valid.append(x_v[index])
        y_valid.append(y_v[index])

    x_train = np.array(x_train)
    x_valid = np.array(x_valid)

    print(x_train.shape, len(y_train))
    print(x_valid.shape, len(y_valid))

    np.save('dogscats_x_train_flattened_200',x_train)
    np.save('dogscats_y_train_flattened_200',y_train)
    np.save('dogscats_x_valid_flattened_50',x_valid)
    np.save('dogscats_y_valid_flattened_50',y_valid)

#get_image_data()

'''
x = r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_x_train.npy',
y = r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_y_train.npy',
x_valid= r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_x_valid.npy',
y_valid = r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_y_valid.npy',
'''
#lr for breast cancer = 0.05
#lr for wine = .0003

runNumpyLogisticRegression( 
                        

                            x = r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_x_train.npy',
                            y = r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_y_train.npy',
                            x_valid= r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_x_valid.npy',
                            y_valid = r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_y_valid.npy',
                            
                            num_iterations = 2000, 
                            learning_rate = .005,
                            title = 'Dog Cats Confusion Matrix',
                            labels = ['cat','dog'])


'''
runPCA( path = '',
        dataset_x=r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_x_train.npy',
        dataset_y= r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_y_train.npy',
        parseData = False,
        standardize = False, 
        type = 'numpy',
        n_components = 2,
        title = 'Wisconsin Breast Cancer PCA (Normalized Data)',
        labels = ['benign','malignant'])
'''
'''
runLDA(         dataset_x=r'NumpyData\Breast Cancer\wdbc\raw\breast_cancer_wdbc_x_train_raw.npy',
                dataset_y= r'NumpyData\Breast Cancer\wdbc\raw\breast_cancer_wdbc_y_train_raw.npy',
                standardize = False,
                type = 'numpy', 
                num_classes=2,
                num_features=30,
                title='Wisconsin Breast Cancer LDA (Normalized Data)',
                labels = ['benign','malignant'])
'''


'''
runPCA( path = '',
        dataset_x=r'NumpyData\Wine\wine_x_train.npy',
        dataset_y= r'NumpyData\Wine\wine_y_train.npy',
        parseData = False,
        standardize = True, 
        type = 'numpy',
        n_components = 2,
        title = 'Wine PCA (Normalized Data)',
        labels = ['class 0','class 1', 'class 2'])
'''
'''
runLDA(         dataset_x=r'NumpyData\Wine\wine_x_train.npy',
                dataset_y= r'NumpyData\Wine\wine_y_train.npy',
                standardize = True,
                type = 'numpy', 
                num_classes=3,
                num_features=13,
                title='Wine LDA (Normalized Data)',
                labels = ['class 0','class 1', 'class 2'])
'''

'''
#for pima indians
alpha = .02
num_hidden_units = 70
num_layers =1
batch_size =200
epochs = 1000


#defaults
alpha = 0.001
num_hidden_units = 12
num_layers = 1
batch_size = 10
epochs = 250

classifiers.load_dataset(
                            r'NumpyData\Pima Indians\pima_indians_x_train.npy',
                            r'NumpyData\Pima Indians\pima_indians_y_train.npy',
                            r'NumpyData\Pima Indians\pima_indians_x_valid.npy',
                            r'NumpyData\Pima Indians\pima_indians_y_valid.npy')
        
result = classifiers.CustomDeepModel(input_size = (8,),
                        num_layers = num_layers, 
                        num_hidden_units = num_hidden_units,
                        num_outputs = 2,
                        output_activation = 'softmax',
                        hidden_activation = 'relu', 
                        loss = 'categorical_crossentropy',
                        optimizer = 'adam',
                        learning_rate = alpha,
                        epochs = epochs, 
                        batch_size = batch_size)

'''

    