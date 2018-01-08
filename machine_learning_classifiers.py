'''
Bryon Kucharski
Wentworth Institute of Technology
Fall 2017

Class with variety of machine learning classifiers from Keras and Scikit

'''

import machine_learning_utils as utils
from numpy_logistic_regression import numpy_logistic_regression
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


    def KerasDeepModel(self, epochs, modelName):
        
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

    def CustomDeepModel(self,input_size,num_layers, num_hidden_units,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate,epochs, batch_size):
        '''
        Assumes data is already preprocessed with mean = 0 and std = 1
        '''
        # x_std = (self.x_train - np.mean(self.x_train,axis=0) ) / np.std(self.x_train,axis=0)
        # x_valid_std = (self.x_valid - np.mean(self.x_valid,axis=0) ) / np.std(self.x_valid,axis=0)

        x_std = self.x_train
        x_valid_std = self.x_valid
        tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

        self.model = utils.custom_Deep_Model(input_size, num_layers, num_hidden_units,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate)
        

        self.y_train = utils.one_hot(self.y_train, num_outputs) 
        self.y_valid = utils.one_hot(self.y_valid, num_outputs)

        #for i in range(num_iterations):
            
        self.model.fit(x_std, self.y_train,validation_data = (x_valid_std, self.y_valid), epochs=epochs, batch_size=batch_size,  callbacks=[tbCallBack])
        scores = self.model.evaluate(self.x_valid, self.y_valid)
        print("Baseline Error: %.2f%%" % (100-scores[1]*100))
        return scores

    def CustomCNNModel(self,input_size, num_layers,num_outputs,output_activation,hidden_activation, loss, optimizer,learning_rate,epochs,batch_size, filter_size, kernal_size, pooling_size):
        '''
        Assumes data is already preprocessed with mean = 0 and std = 1
        '''

        #  x_std = (self.x_train - np.mean(self.x_train,axis=0) ) / np.std(self.x_train,axis=0)
        #  x_valid_std = (self.x_valid - np.mean(self.x_valid,axis=0) ) / np.std(self.x_valid,axis=0)
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
        http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
        '''
        if(standardize):
            x_std = (self.x_train - np.mean(self.x_train,axis=0) ) / np.std(self.x_train,axis=0)
        else:
            x_std = self.x_train
        #x_std = self.x_train
 
        #x_std = self.x_train
   
        #x_std = self.x_train

        cov = np.cov(x_std.T)
        #print('cov',cov)
        
        evals, evecs = np.linalg.eig(cov)
   
        # sort eigenvalue in decreasing order
        idx = np.argsort(abs(evals))[::-1]
        evecs = evecs[:,idx]
  

        #get the top eigenvalues
        evals = evals[idx]
        #get the top eigenvectors
       
        #print('W',W)
        #print(evecs)
        pca = x_std.dot(evecs)
        pca = pca[:,:n_components]

        utils.plotPCA(pca,self.y_train, n_components,labels, xlabel = 'Principal Component 1', ylabel = 'Principal Component 2', title = plot_title)

        return pca

    def scikit_PCA(self, n_components,labels, title = ''):
        model = sklearnPCA(n_components=n_components)
        pca = model.fit_transform(self.x_train)
        utils.plotPCA(pca,self.y_train, n_components,labels, title = title)
        return pca

    def scikit_LDA(self, num_classes,labels, plot = True,predict_image='None'):
        model = LinearDiscriminantAnalysis()
        lda = model.fit_transform(self.x_train,self.y_train)
        #acc = model.score(self.x_valid, self.y_valid)
       # print("LDA Accuracy: {:.2f}%".format(acc * 100))
        if plot:
            utils.plotPCA(lda,self.y_train, num_classes-1, labels)
        return lda, model

    def numpy_LDA(self, num_classes, num_features, labels, standardize= True , title = ''):
        '''
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

    def test(self):
   

        #self.x_train, self.y_train = utils.load_dataset(r'C:\Users\kucharskib\Desktop\x_scikit_raw.npy' , r'C:\Users\kucharskib\Desktop\y_scikit_raw.npy')
        self.x_train, self.y_train = utils.load_dataset(r'x_valid_scikit_raw.npy' , r'y_valid_scikit_raw.npy')

        #self.x_valid, self.y_valid = utils.load_dataset(r'x_valid_scikit_raw.npy' , r'y_valid_scikit_raw.npy')
        lda = self.scikit_LDA(num_classes = 2, plot = False)

        class0 = lda[self.y_train == 0]
        class1 = lda[self.y_train == 1]

        '''
        m1 = np.median(class1)
        m0 = np.median(class0)

        s1 = np.std(class1)
        s0 = np.std(class0)


        thres = ((s0+m0) + (m1-s1)) / 2
        '''
        thres = -0.023
        class0correct = 0
        class1correct = 0
       

        for i in range(len(lda)):
            if(lda[i] > thres) and (self.y_train[i] == 1):
                class1correct = class1correct + 1
            elif(lda[i] < thres) and (self.y_train[i] == 0):
                class0correct = class0correct + 1

        print('class0 size',class0.shape)
        print('class1 size ',class1.shape)
        '''
        print('class0 median, ', m0)
        print('class1 median, ', m1)

        print('class0 std, ', s0)
        print('class1 std, ', m1)

        print('thres', thres)
        '''
        print('class0 correct',class0correct)
        print('class1 correct ',class1correct)

        print('class0 accuracy',(class0correct/len(class0)) * 100)
        print('class1 accuracy ',(class1correct/len(class1)) * 100)

    def wine_test(self):
        
        self.x_train, self.y_train = utils.load_dataset(r'NumpyData\wine\wine.data.txt_x.npy' , r'NumpyData\wine\wine.data.txt_y.npy')
        self.x_valid, self.y_valid = utils.load_dataset(r'NumpyData\wine\wine.data.txt_x_valid.npy' , r'NumpyData\wine\wine.data.txt_y_valid.npy')

        results, model = self.scikit_LDA(num_classes=3, plot = False)

        predictions = []
        for input in self.x_valid:
            prediction = model.predict(input)
            predictions.append(prediction)

        num_correct = 0
        for i in range(len(predictions)):
            if predictions[i] == self.y_valid[i]:
                num_correct += 1
            print('Pred: ', predictions[i], 'Actual ', self.y_valid[i])
        
        print('Accuracy: ',  num_correct / len(predictions) * 100)

        #self.get_thres(results, 2,3)

    def get_thres(self,lda, num_axis, num_classes):
        
    
        classes = []
        for i in range(num_classes):
            classes.append(lda[:,0][self.y_train == i])

        medians = []
        stds = []
        for i in range(classes):
            medians.append(np.medians(classes[i]))
            stds.append(mp.std(classes[i]))

        thres = ()+() / num_classes
        

        
        m1 = np.median(class1)
        m0 = np.median(class0)

        s1 = np.std(class1)
        s0 = np.std(class0)


        thres = ((s0+m0) + (m1-s1)) / 2
        
        
            
        

        
