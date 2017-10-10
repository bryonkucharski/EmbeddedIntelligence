from machine_learning_classifiers import machine_learning_classifiers

classifiers = machine_learning_classifiers()



#classifiers.parse_dataset('dogscats/train', 'jpg', 'dogscats_x_train_preprocessed_vgg', 'dogscats_y_train_preprocessed_vgg',224,'keras')
#classifiers.parse_dataset('dogscats/valid','jpg', 'dogscats_x_valid_preprocessed_vgg', 'dogscats_y_valid_preprocessed_vgg',224, 'keras')
classifiers.load_dataset('dogscats_x_train_preprocessed.npy', 'dogscats_y_train_preprocessed.npy', 'dogscats_x_valid_preprocessed.npy', 'dogscats_y_valid_preprocessed.npy')
classifiers.SimpleCNN(10,'dogscats_simpledeepmodel_preprocessed.h5')

#classifiers.parse_dataset('dogscats/train', 'jpg', 'x_scikit_preprocessed', 'y_scikit_preprocessed',50,'scikit')
#classifiers.parse_dataset('dogscats/valid', 'jpg', 'x_test_scikit_preprocessed', 'y_test_scikit_preprocessed',50,'scikit')

#classifiers.load_dataset('x_scikit_preprocessed.npy', 'y_scikit_preprocessed.npy', 'x_test_scikit_preprocessed.npy', 'y_test_scikit_preprocessed.npy')


#print('Running Logistic Regression')
#classifiers.LogisticRegression()

print('Running LinearSVM')
classifiers.LinearSVM()

print('Running KNN')
classifiers.KNN()

print('Running Random Forest')
classifiers.RandomForestClassifier()

print('Running GaussianNB')
classifiers.GaussianNB()


