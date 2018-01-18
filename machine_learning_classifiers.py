'''
Bryon Kucharski
Wentworth Institute of Technology
Fall 2017

Class with variety of machine learning classifiers from Keras, Scitkit, and Numpy
'''

import machine_learning_utils as utils
from numpy_logistic_regression import numpy_logistic_regression
from numpy_artificial_neural_network import numpy_artificial_neural_network
import numpy as np
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot
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
        self.model = None

    def load_dataset(self,x_train_name,y_train_name,x_valid_name,y_valid_name):
        '''
            loads a previously parsed numpy array
        '''
        self.x_train, self.y_train = utils.load_dataset(x_train_name,y_train_name)
        self.x_valid, self.y_valid = utils.load_dataset(x_valid_name,y_valid_name)
       
    
    def load_vector_dataset(self,x_name, y_name):
        '''
            loads a previously parsed numpy array
        '''
        self.x_train, self.y_train = utils.load_dataset(x_name, y_name)
        self.x_valid = None
        self.y_valid = None


    def KerasDeepModel(self, num_classes, IMG_SIZE,epochs, modelName, saveModel = 'False'):
        
        self.model = utils.Keras_Website_Model(num_classes, IMG_SIZE)
        self.y_train = utils.one_hot(self.y_train, num_classes) 

        self.y_valid = utils.one_hot(self.y_valid, num_classes)
        self.model.fit(self.x_train, self.y_train, validation_data = (self.x_valid, self.y_valid), epochs=epochs, batch_size=200, verbose=2)
        scores = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
        if saveModel:
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

    def CustomDeepModel(self,input_size,num_layers, num_hidden_units,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate,epochs, batch_size):
        '''
        Assumes data is already preprocessed with mean = 0 and std = 1
        '''

        x_std = self.x_train
        x_valid_std = self.x_valid
        tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

        self.model = utils.custom_Deep_Model(input_size, num_layers, num_hidden_units,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate)
        

        self.y_train = utils.one_hot(self.y_train, num_outputs) 
        self.y_valid = utils.one_hot(self.y_valid, num_outputs)
            
        self.model.fit(x_std, self.y_train,validation_data = (x_valid_std, self.y_valid), epochs=epochs, batch_size=batch_size,  callbacks=[tbCallBack])
        scores = self.model.evaluate(self.x_valid, self.y_valid)
        print("Baseline Error: %.2f%%" % (100-scores[1]*100))
        return scores

    def CustomCNNModel(self,input_size, num_layers,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate,epochs,batch_size, filter_size, kernal_size, pooling_size):
        '''
        Assumes data is already preprocessed with mean = 0 and std = 1
        '''

        x_std = self.x_train
        x_valid_std = self.x_valid

        self.model = utils.custom_CNN_Model(input_size, num_layers,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate, filter_size, kernal_size, pooling_size)

        self.y_train = utils.one_hot(self.y_train, num_outputs) 
        self.y_valid = utils.one_hot(self.y_valid, num_outputs)

        self.model.fit(x_std, self.y_train,validation_data = (x_valid_std, self.y_valid), epochs=epochs, batch_size=batch_size)

        scores = self.model.evaluate(self.x_valid, self.y_valid)
        print(self.model.summary())

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

    def numpy_PCA(self, n_components,labels, standardize = True, plot_title = ''):
        '''
        numpy implementation of PCA
        http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
        '''
        if(standardize):
            x_std = (self.x_train - np.mean(self.x_train,axis=0) ) / np.std(self.x_train,axis=0)
        else:
            x_std = self.x_train

        cov = np.cov(x_std.T)
        #print('cov',cov)
        
        evals, evecs = np.linalg.eig(cov)
   
        # sort eigenvalue in decreasing order
        idx = np.argsort(abs(evals))[::-1]

        #get the top eigenvectors
        evecs = evecs[:,idx]
  
        #get the top eigenvalues
        evals = evals[idx]
       
        pca = x_std.dot(evecs)
        pca = pca[:,:n_components]

        utils.plotPCA(pca,self.y_train, n_components,labels, xlabel = 'Principal Component 1', ylabel = 'Principal Component 2', title = plot_title)

        return pca

    def scikit_PCA(self, n_components,labels, title = ''):
        '''
        scikit implementation of PCA
        '''
        model = sklearnPCA(n_components=n_components)
        pca = model.fit_transform(self.x_train)
        utils.plotPCA(pca,self.y_train, n_components,labels, title = title)
        return pca

    def scikit_LDA(self, num_classes,labels, plot = True,predict_image='None'):
        '''
        scikit implementation of LDA
        '''
        model = LinearDiscriminantAnalysis()
        lda = model.fit_transform(self.x_train,self.y_train)
        #acc = model.score(self.x_valid, self.y_valid)
       # print("LDA Accuracy: {:.2f}%".format(acc * 100))
        if plot:
            utils.plotPCA(lda,self.y_train, num_classes-1, labels)
        return lda, model

    def numpy_LDA(self, num_classes, num_features, labels, standardize= True , title = ''):
        '''
        numpy implementation of LDA
        http://sebastianraschka.com/Articles/2014_python_lda.html
        
            1) d-dimensional mean vectors, (num_classes,num_features)
            2) Sw - within-class scatter matrix (num_features,num_features)
            3) Sb - Between-class scatter matrix (num_features,num_features)
            4) Sw^-1*Sb - find eig
            5)Y = W*X (num_input,num_features)

        '''
        if(standardize):
            x_std = (self.x_train - np.mean(self.x_train,axis=0) ) / np.std(self.x_train,axis=0)
        else:
            x_std = self.x_train
        print('numpy lda statistics:')
        print('size', x_std.shape)
        print('mean', np.mean(x_std))
        print('std', np.std(x_std))
        print('Step 1) d-dimensional mean vectors ')
        #1)  d-dimensional mean vectors
        d = []
    

        #NOTE: this assumes index of y starts at 0
        for i in range(0,num_classes):        
            mean_class_i = np.mean(x_std[self.y_train==i],axis=0) # shorthand way of saying index of x_train where index of y_train == i
            d.append(mean_class_i)
        
        print('Step 2) Sw - within class scatter matrix ')
        #2) Sw - within class scatter matrix
        Sw = np.zeros((num_features,num_features))

        #iterate through classes
        for class_index in range(len(d)):
            print('class_index: ', class_index)
            current_matrix = np.zeros((num_features,num_features))
            #iterate through each datapoint and compute the scatter matrix
            for input_data in x_std[self.y_train == (class_index)]:
                #reshape data
                input_data = input_data.reshape(num_features,1)
                mean_vec = d[class_index].reshape(num_features,1)

                #update scatter matrix of current class
                current_matrix += (input_data-mean_vec).dot((input_data-mean_vec).T)

            #update overall scatter matrix
            Sw += current_matrix
        print('Step 3) Sb - Between-class scatter matrix')
        #3)Sb
        overall_mean = np.mean(x_std, axis=0)
       
        Sb = np.zeros((num_features,num_features))

        #iterate through classes
        for class_index in range(len(d)):
            
            #find number of data points in that class
            n = x_std[self.y_train==(class_index )].shape[0]

            #reshape data
            mean_vec = d[class_index].reshape(num_features,1)
            overall_mean = overall_mean.reshape(num_features,1)

            Sb += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        print('Step 4) Sw^-1*Sb')
        #4) Sw^-1*Sb - find eig

        linear_discriminants = np.linalg.inv(Sw).dot(Sb)

        evals, evecs = np.linalg.eig(linear_discriminants)
    
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        #get the top eigenvalues
        evals = evals[idx]
        #get the top eigenvectors
        W = evecs[:,:num_classes-1]

        print('Step 5) Transform Data')
        #5) Transform data
        lda = (x_std.dot(W)) *-1  #-1 to match scikit output
     
        utils.plotPCA(lda,self.y_train, 0,labels, title = title, xlabel = 'Linear Discriminant 1', ylabel = 'Linear Discriminant 2')
    
        return lda

    def numpy_logistic_reg(self, num_iterations, learning_rate, showPlot = True, confusion_title = '', confusion_labels = ''):
        
        x  = self.x_train
        y = self.y_train
        x_valid = self.x_valid
        y_valid = self.y_valid

        lr = numpy_logistic_regression()

        costs, iterations,y_pred = lr.fit(x.T,y,x_valid.T,y_valid, num_iterations, learning_rate)

        print( 'Train Accuracy: ', lr.get_parameters()['train accuracy'], '%')
        print( 'Test Accuracy: ', lr.get_parameters()['test accuracy'], '%')

        if(showPlot):
            axes = pyplot.gca()
            axes.set_ylim(0.0,1.0)
            pyplot.plot(iterations,costs)
            pyplot.title('Logistic Regression Cost')
            pyplot.xlabel('Iteration')
            pyplot.ylabel('Cost')
            pyplot.show()

        utils.get_confusion_matrix(y_valid,y_pred.reshape(len(y_valid)), labels = confusion_labels, title = confusion_title)
    
    def numpy_neural_net(self,dims, lr, num_iterations):
        nn = numpy_artificial_neural_network()
        
        self.x_train=np.swapaxes(self.x_train,0,1)
        self.y_train = np.reshape(self.y_train,(1,len(self.y_train)))

        
        self.x_valid = np.swapaxes(self.x_valid,0,1)
        self.y_valid = np.reshape(self.y_valid,(1,len(self.y_valid)))


        nn.fit(
                X = self.x_train,
                Y = self.y_train,
                X_valid = self.x_valid,
                Y_valid = self.y_valid ,
                layers_dims = dims,
                learning_rate = lr,
                num_iterations=num_iterations
            )

        
