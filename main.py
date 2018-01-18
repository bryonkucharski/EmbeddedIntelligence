'''

Bryon Kucharski
Wentworth Institute of Technology
Fall 2017

'''

from machine_learning_classifiers import machine_learning_classifiers
import machine_learning_utils as utils
import numpy as np


classifiers = machine_learning_classifiers()

def runKeras(x,y,x_valid,y_valid,num_classes, IMG_SIZE, epochs):
    
    classifiers.load_dataset(x, y ,x_valid, y_valid)
    classifiers.KerasDeepModel(num_classes = num_classes, IMG_SIZE = IMG_SIZE, epochs = epochs)


def runSciKit(x,y,x_valid,y_valid):

    classifiers.load_dataset(x, y ,x_valid, y_valid)

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

def runPCA( dataset_x, dataset_y,parseData = False,standardize = True, type = 'scikit', n_components = 2, title = '', labels = [], path = '',):
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

def runNumpyLogisticRegression(x,y,x_valid, y_valid, num_iterations, learning_rate, title, labels):
    
    classifiers.load_dataset(x,y,x_valid,y_valid)
                      
    classifiers.numpy_logistic_reg(num_iterations, learning_rate, confusion_title=title, confusion_labels=labels)

def runNumpyDeepNeuralNetwork(x,y,x_valid, y_valid, num_iterations, learning_rate,dims):
    
    classifiers.load_dataset(x,y,x_valid,y_valid)

    classifiers.numpy_neural_net(dims = dims , lr = learning_rate, num_iterations = num_iterations)

def runCustomDeepModel(x,y,x_valid,y_valid,input_size, num_layers, num_hidden_units, num_outputs, output_activation, hidden_activation, loss, optimizer, learning_rate, epochs,batch_size ):
    
    classifiers.load_dataset(x,y,x_valid,y_valid)
                            
        
    result = classifiers.CustomDeepModel(input_size = input_size,
                        num_layers = num_layers, 
                        num_hidden_units = num_hidden_units,
                        num_outputs = num_outputs,
                        output_activation = 'softmax',
                        hidden_activation = 'relu', 
                        loss = 'categorical_crossentropy',
                        optimizer = 'adam',
                        learning_rate = learning_rate,
                        epochs = epochs, 
                        batch_size = batch_size)
    return result

'''
runNumpyDeepNeuralNetwork(
        x = r'NumpyData\Dogscats\Flattened\Subset 200\dogscats_x_train_flattened_200.npy',
        y = r'NumpyData\Dogscats\Flattened\Subset 200\dogscats_y_train_flattened_200.npy' ,
        x_valid = r'NumpyData\Dogscats\Flattened\Subset 200\dogscats_x_valid_flattened_50.npy',
        y_valid = r'NumpyData\Dogscats\Flattened\Subset 200\dogscats_y_valid_flattened_50.npy',  
        num_iterations = 1000, 
        learning_rate = 0.009,
        dims = [12288,7,1]
        )
'''
'''
runNumpyLogisticRegression(
        x = r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_x_train.npy',
        y = r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_y_train.npy',
        x_valid= r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_x_valid.npy',
        y_valid = r'NumpyData\Breast Cancer\wdbc\preprocessed\breast_cancer_wdbc_y_valid.npy',
        
        num_iterations = 2000, 
        learning_rate = .005,
        title = 'Dog Cats Confusion Matrix',
        labels = ['cat','dog']
        )
'''
'''
runPCA( 
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
runPCA( 
        dataset_x=r'NumpyData\Wine\preprocessed\wine_x_train.npy',
        dataset_y= r'NumpyData\Wine\preprocessed\wine_y_train.npy',
        parseData = False,
        standardize = True, 
        type = 'numpy',
        n_components = 2,
        title = 'Wine PCA (Normalized Data)',
        labels = ['class 0','class 1', 'class 2'])
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
runLDA(         dataset_x=r'NumpyData\Wine\wine_x_train.npy',
                dataset_y= r'NumpyData\Wine\wine_y_train.npy',
                standardize = True,
                type = 'numpy', 
                num_classes=3,
                num_features=13,
                title='Wine LDA (Normalized Data)',
                labels = ['class 0','class 1', 'class 2'])
'''

#Should be done on a GPU
runKeras(
        x = r'NumpyData\Dogscats\3D\dogscats_x_train_raw.npy',
        y = r'NumpyData\Dogscats\3D\dogscats_y_train_raw.npy',
        x_valid = r'NumpyData\Dogscats\3D\dogscats_x_valid_raw.npy' ,
        y_valid = r'NumpyData\Dogscats\3D\dogscats_y_valid_raw.npy',
        num_classes = 2,
        IMG_SIZE= 224,
        epochs = 30,
      
        )

'''
#this did not work too well
runSciKit(
            x = r'NumpyData\Dogscats\Flattened\Subset 200\dogscats_x_train_flattened_200.npy',
            y = r'NumpyData\Dogscats\Flattened\Subset 200\dogscats_y_train_flattened_200.npy',
            x_valid = r'NumpyData\Dogscats\Flattened\Subset 200\dogscats_x_valid_flattened_50.npy' ,
            y_valid = r'NumpyData\Dogscats\Flattened\Subset 200\dogscats_y_valid_flattened_50.npy'
            )
'''
'''
#for pima indians
alpha = .02
num_hidden_units = 70
num_layers =1
batch_size =200
epochs = 1000
'''
'''
#defaults
alpha = 0.001
num_hidden_units = 12
num_layers = 1
batch_size = 10
epochs = 250
'''
'''
runCustomDeepModel(     x = r'NumpyData\Pima Indians\pima_indians_x_train.npy',
                        y = r'NumpyData\Pima Indians\pima_indians_y_train.npy',
                        x_valid = r'NumpyData\Pima Indians\pima_indians_x_valid.npy',
                        y_valid = r'NumpyData\Pima Indians\pima_indians_y_valid.npy',
                        input_size = (8,),
                        num_layers = 1, 
                        num_hidden_units = 70,
                        num_outputs = 2,
                        output_activation = 'softmax',
                        hidden_activation = 'relu', 
                        loss = 'categorical_crossentropy',
                        optimizer = 'adam',
                        learning_rate = .02,
                        epochs = 1000, 
                        batch_size = 200)
'''

    