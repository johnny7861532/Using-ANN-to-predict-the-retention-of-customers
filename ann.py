# Artificial Neural Network
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#select the row we need from 2 to 12 
#but 3-13 
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data from string to Int for training
# And our dataset only have 2. one is country, and the other is gender.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#set labelecoder to X_1 country 
labelencoder_X_1 = LabelEncoder()
#set row1 and turn from france, spain, german into 0,1,2
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#set labelecoder to X_2 gender
labelencoder_X_2 = LabelEncoder()
#set row2 and turn from male and female into 0,1
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
#create dummy variable for balence the oder of country or 3 might be the most import variable but ir isnt 
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# we don;t need 3 dummy variable to represent country, becasue 0,0,1 = 0,0
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# to normalize data or some feature might dominate other's by it's value
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#start to build ANN for training
import keras
#initalize neron network
from keras.models import Sequential
#build layers
from keras.layers import Dense
#import a model to prevet from overfit
from keras.layers import Dropout   
# initalize ANN
classifier = Sequential()
#add the input layer and first hidden layer with dropout
#init represent weight and 'uniform' represent random number for weight
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
classifier.add(Dropout(p = 0.1))
#add second hidden layer and because we have already make input_dim at first layer with dropout
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
classifier.add(Dropout(p = 0.1))
#output layer because we only need 1 or 0 we need only one node
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
#compiling the ANN
#optimizer = the algorithm to modify the weight when backward
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#fit ANN to training set
#batch size = how many data set train and then turn the weight, and epochs is how many time did this set train
classifier.fit(X_train,y_train,batch_size = 10,epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# setting the threhold for the 1,0 result for confusion matrix and we set 0.5
y_pred = (y_pred > 0.5)

#homework predict if the traget will leave the bank
#[[]] means one 1 horrizen data in array
target_pred = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
target_pred = (target_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#trying to tune and improve the ANN performance
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
#initalize neron network
from keras.models import Sequential
#build layers
from keras.layers import Dense  
#create layer and node inside the function call build_classifier (python)
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


#using the k_fold cross validation can be more effect on training model
classifier = KerasClassifier(build_fn = build_classifier,batch_size = 10,epochs = 100)
#cv = cross val and no_job = how many cpu u use -1 for all cpu
accuraccies = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10, n_jobs = -1 )

mean = accuraccies.mean()
variance = accuraccies.std()

#find the best paramete for our neuron network

#trying to tune and improve the ANN performance
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
#initalize neron network
from keras.models import Sequential
#build layers
from keras.layers import Dense  
#create layer and node inside the function call build_classifier (python)
def build_classifier(optimizer,activation,outputactivation):
    classifier = Sequential()
    classifier.add(Dense(activation=activation, input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation=activation, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation=outputactivation, units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


#using the k_fold cross validation can be more effect on training model
classifier = KerasClassifier(build_fn = build_classifier)
#dictionary in python is {}
parameters = {'batch_size':[25, 32],
              'epochs':[100, 500],
              'optimizer':['adam', 'rmsprop'],
              'activation':['relu','sigmoid','softplus','tanh'],
              'outputactivation':['relu','sigmoid','softplus','tanh']
              }
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv =10)

grid_search = grid_search.fit(X_train,y_train)

best_parameter =  grid_search.best_params_
best_accuracy =  grid_search.best_score_

#my best score formular
#trying to tune and improve the ANN performance
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
#initalize neron network
from keras.models import Sequential
#build layers
from keras.layers import Dense  

from keras.layers.advanced_activations import LeakyReLU
#you need to add advanced_activation then call out LeakyRelu or PreLu for activation function
leakyrelu = LeakyReLU(alpha=0.3)
#create leakyrelu parameter
#create layer and node inside the function call build_classifier (python)
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation=leakyrelu, input_dim=11, units=11, kernel_initializer="uniform"))
    classifier.add(Dense(activation=leakyrelu, units=22, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


#using the k_fold cross validation can be more effect on training model
classifier = KerasClassifier(build_fn = build_classifier,batch_size = 24,epochs = 500)
#cv = cross val and no_job = how many cpu u use -1 for all cpu
accuraccies = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10, n_jobs = 1 )

mean = accuraccies.mean()
variance = accuraccies.std()
target_pred = build_classifier().predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
target_pred = (target_pred > 0.5)


from keras.models import load_model 
build_classifier().save_weights('MyConvNets.h5')

model = load_model('my_model.h5')







