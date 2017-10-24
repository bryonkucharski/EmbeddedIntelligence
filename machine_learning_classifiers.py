'''
Bryon Kucharski
Wentworth Institute of Technology
Fall 2017

Class with variety of machine learning classifiers from Keras and Scikit

'''

import machine_learning_utils as utils
import numpy as np
from sklearn.metrics import classification_report

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
np.set_printoptions(threshold=np.nan)


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
        utils.get_dataset(path, extension, x_name, y_name,IMG_SIZE, modelType)
    
    def load_dataset(self,x_train_name,y_train_name,x_valid_name,y_valid_name):
        '''
            loads a previously parsed numpy array
        '''
        self.x_train, self.y_train,self.x_valid,self.y_valid = utils.load_dataset(x_train_name,y_train_name,x_valid_name,y_valid_name)
        print("-Sizes-\nx_train: ",self.x_train.shape, '\ny_train ', len(self.y_train) , '\nx_valid ' ,self.x_valid.shape , '\ny_valid ' , len(self.y_valid))

    def parse_vector_dataset(self,path, x_name, y_name):
        '''
        parses dataset from file of vectors into numpy array
        '''
        utils.get_vector_dataset(path, x_name, y_name)
    
    def load_vector_dataset(self,x_name, y_name):
        '''
            loads a previously parsed numpy array
        '''
        self.x_train, self.y_train = utils.load_vector_dataset(x_name, y_name)
        self.x_valid = None
        self.y_valid = None


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

    def numpy_PCA(self, n_components):
        x_std = self.x_train
        cov = np.cov(x_std.T)
  
        evals, evecs = np.linalg.eig(cov)
      
        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]

        #get the top eigenvalues
        evals = evals[idx]
        #get the top eigenvectors
        W = evecs[:,:n_components]

        pca = x_std.dot(W)
    
        utils.plotPCA(pca,self.y_train)

        return pca

    def scikit_PCA(self, n_components):
        model = sklearnPCA(n_components=n_components)
        pca = model.fit_transform(self.x_train)
        utils.plotPCA(pca,self.y_train)
        return pca

    def scikit_LDA(self):
        model = LinearDiscriminantAnalysis()
        lda = model.fit_transform(self.x_train,self.y_train) 
        utils.plotPCA(lda,self.y_train)
        return lda

    def numpy_LDA(self,n_components, num_classes, num_features):
        '''
            1) d-dimensional mean vectors, (num_classes,num_features)
            2) Sw - within-class scatter matrix (num_features,num_features)
            3) Sb - Between-class scatter matrix (num_features,num_features)
            4) Sw^-1*Sb - find eig
            5)Y = W*X (num_input,num_features)

        '''
        #1)  d-dimensional mean vectors
        d = []
        for i in range(1,num_classes+1):
            mean_class_i = np.mean(self.x_train[self.y_train==i],axis=0) # shorthand way of saying index of x_train where index of y_train == i
            d.append(mean_class_i)

        
        #2) Sw - within class scatter matrix
        Sw = np.zeros((num_features,num_features))

        #iterate through classes
        for class_index in range(len(d)):
           
            
            current_matrix = np.zeros((num_features,num_features))
            #iterate through each datapoint and compute the scatter matrix
            for input_data in self.x_train[self.y_train == (class_index + 1)]:
                
                #reshape data
                input_data = input_data.reshape(num_features,1)
                mean_vec = d[class_index].reshape(num_features,1)

                #update scatter matrix of current class
                current_matrix += (input_data-mean_vec).dot((input_data-mean_vec).T)

            #update overall scatter matrix
            Sw += current_matrix

        #3)Sb
        overall_mean = np.mean(self.x_train, axis=0)

        Sb = np.zeros((num_features,num_features))

        #iterate through classes
        for class_index in range(len(d)):
            
            #find number of data points in that class
            n = self.x_train[self.y_train==(class_index + 1)].shape[0]

            #reshape data
            mean_vec = d[class_index].reshape(num_features,1)
            overall_mean = overall_mean.reshape(num_features,1)

            Sb += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

        #4) Sw^-1*Sb - find eig

        linear_discriminants = np.linalg.inv(Sw).dot(Sb)

        evals, evecs = np.linalg.eig(linear_discriminants)
    
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        #get the top eigenvalues
        evals = evals[idx]
        #get the top eigenvectors
        W = evecs[:,:n_components]

        #5) Transform data
        lda = (self.x_train.dot(W)) *-1  #-1 to match scikit output

        utils.plotPCA(lda,self.y_train)
    
        return lda


    def showImage(self,img):
        utils.showImage(img)
    
    def printImage(self,index):
        print(self.x_train[index])

    def test(self,path, extension,IMG_SIZE, modelType = 'keras'):
        utils.get_data(path, extension,IMG_SIZE, modelType = 'keras')
