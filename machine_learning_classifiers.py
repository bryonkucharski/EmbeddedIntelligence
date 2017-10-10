'''
Bryon Kucharski
Wentworth Institute of Technology
Fall 2017

Class with variety of machine learning classifiers from Keras and Scikit

'''

import machine_learning_utils as utils
from sklearn.metrics import classification_report

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class machine_learning_classifiers:
    '''
        Variety of machine learning classifiers from Keras and Scikit
    '''
    def __init__(self):
        self.x_train = []
        self.y_train= []
        self.x_valid = []
        self.y_valid = []

    def parse_dataset(self,path, extension, x_name, y_name,IMG_SIZE, modelType = 'keras'):
        '''
            parses data form a folder into a numpy array
        '''
        utils.parse_dataset(path, extension, x_name, y_name,IMG_SIZE, modelType)
    
    def load_dataset(self,x_train_name,y_train_name,x_valid_name,y_valid_name):
        '''
            loads a previously parsed numpy array
        '''
        self.x_train, self.y_train,self.x_valid,self.y_valid = utils.load_dataset(x_train_name,y_train_name,x_valid_name,y_valid_name)
        print("-Sizes-\nx_train: ",self.x_train.shape, '\ny_train ', len(self.y_train) , '\nx_valid ' ,self.x_valid.shape , 'y_valid ' , len(self.y_valid))

    def DeepModel(self, epochs, modelName):
        
        self.model = utils.Keras_Website_Model(2, 224)
        self.y_train = utils.one_hot(self.y_train, 2) 

        self.y_valid = utils.one_hot(self.y_valid, 2)
        self.model.fit(self.x_train, self.y_train, validation_data = (self.x_valid, self.y_valid), epochs=epochs, batch_size=200, verbose=2)
        scores = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
        self.model.save_weights(modelName)
        print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    
    def SimpleCNN(self, epochs, modelName):
        self.model = utils.CNN_deepModel(2, 224)
        self.y_train = utils.one_hot(self.y_train, 2) 
        self.y_valid = utils.one_hot(self.y_valid, 2)
        self.model.fit(self.x_train, self.y_train, validation_data = (self.x_valid, self.y_valid), epochs=epochs, batch_size=200, verbose=2)
        scores = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
        self.model.save_weights(modelName)
        print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    def LinearSVM(self):

        model = LinearSVC()
        model.fit(self.x_train, self.y_train)
        acc = model.score(self.x_valid, self.y_valid)
        print("SVM Accuracy: {:.2f}%".format(acc * 100))

    def KNN(self):

        model = KNeighborsClassifier()
        model.fit(self.x_train, self.y_train)
        acc = model.score(self.x_valid, self.y_valid)
        print("KNN Accuracy: {:.2f}%".format(acc * 100))

    def LogisticRegression(self):
        model = LogisticRegression()
        model = model.fit(self.x_train, self.y_train)
        acc = model.score(self.x_valid, self.y_valid)
        print("Logistic Regression Accuracy: {:.2f}%".format(acc * 100))

    def GaussianNB(self):
        model = GaussianNB()
        model = model.fit(self.x_train, self.y_train)
        acc = model.score(self.x_valid, self.y_valid)
        print("Gaussian NB Accuracy: {:.2f}%".format(acc * 100))

    def RandomForestClassifier(self):
        model = RandomForestClassifier()
        model = model.fit(self.x_train, self.y_train)
        acc = model.score(self.x_valid, self.y_valid)
        print("Random Forest Accuracy: {:.2f}%".format(acc * 100))

    def showImage(self,img):
        utils.showImage(img)
    
    def printImage(self,index):
        print(self.x_train[index])

    def test(self,path, extension,IMG_SIZE, modelType = 'keras'):
        utils.get_data(path, extension,IMG_SIZE, modelType = 'keras')
