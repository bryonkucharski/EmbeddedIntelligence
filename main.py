from machine_learning_classifiers import machine_learning_classifiers

classifiers = machine_learning_classifiers()

def runKeras(parseData=False):
    
    #to run keras model
    if parseData:
        classifiers.parse_dataset('dogscats/train', 'jpg', 'dogscats_x_train_preprocessed', 'dogscats_y_train_preprocessed',224,'keras')
        classifiers.parse_dataset('dogscats/valid','jpg', 'dogscats_x_valid_preprocessed', 'dogscats_y_valid_preprocessed',224, 'keras')

    classifiers.load_dataset(r'dogscats_x_train_preprocessed.npy', r'dogscats_y_train_preprocessed.npy', r'dogscats_x_valid_preprocessed.npy', r'dogscats_y_valid_preprocessed.npy')
    classifiers.DeepModel(30,'dogscats_deepmodel_preprocessed.h5')


def runSciKit(parseData = False):

    if parseData:
        classifiers.parse_dataset('dogscats/train', 'jpg', 'x_scikit_preprocessed', 'y_scikit_preprocessed',50,'scikit')
        classifiers.parse_dataset('dogscats/valid', 'jpg', 'x_test_scikit_preprocessed', 'y_test_scikit_preprocessed',50,'scikit')
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

def runPCA(dataset,parseData = False, type = 'scikit', n_components = 2):
    if parseData:
        classifiers.parse_vector_dataset(dataset,dataset+"_x", dataset+"_y")

    classifiers.load_vector_dataset(dataset+"_x.npy", dataset+"_y.npy")
    #classifiers.numpy_PCA()
    if type == 'scikit':
        results = classifiers.scikit_PCA(n_components)
    elif type == 'numpy':
        results = classifiers.numpy_PCA(n_components)
    print(results)

def runLDA(dataset,parseData = False, type = 'scikit', n_components = 2, num_classes=3, num_features=13):
    if parseData:
        classifiers.parse_vector_dataset(dataset,dataset+"_x", dataset+"_y")

    classifiers.load_vector_dataset(dataset+"_x.npy", dataset+"_y.npy")
    if type == 'scikit':
        results = classifiers.scikit_LDA()
    elif type == 'numpy':
        results = classifiers.numpy_LDA(n_components, num_classes, num_features)
    print(results)

runLDA('wine.data.txt',False,'scikit')

        
        
