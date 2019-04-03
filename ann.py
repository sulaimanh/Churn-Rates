# The bank will use this model, that we just validated on the test set, on the customers of the bank
# The bank will then be able to apply the model on all the customers of the bank and by ranking the probabilities from the 
#       highest to the lowest, it gets a ranking of the customers most likely to leave the bank. 

# This ANN creates a lot of added value to the bank because by targeting these customers most likely to leave the bank, the bank itself can take some
#       measures to prevent these customers from leaving.


# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Dummy Variable Trap
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# A hyperparameter is a parameter whose value is used to control the learning process. 
# Hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm.
#   This can be simply thought of as the process which one goes through in which they optimize the parameters that impact the model in order
#       to enable the algorithm to perform the best.
from keras.wrappers.scikit_learn import KerasClassifier
# The grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the param_grid parameter.
from sklearn.model_selection import GridSearchCV 
# This is required to initialize the neural network
from keras.models import Sequential
# This is required in order to build the layers of the ANN
from keras.layers import Dense



# This function builds the ANN classifier.
def build_classifier(optimizer):
    # We are going to define it as a sequence of layers
    # This is the neural network that will have a role of classifier because our problem is a classification problem where we have to
    #   predict a class
    classifier = Sequential()
    # We are now ready to start adding layers to this ANN.
    # Adding the input layer and the first hidden layer
    # For now, we are just going to take the average of the number of nodes in the input layer and the number of nodes in the output layer. 
    # Input layer Nodes = 11
    # Output layer Nodes = 1
    # 11+1/2 = 6
    # Arguments:
    #   1st: This is the number of nodes in the hidden layer
    #   2nd: Here, we are randomly initializing the weights close to zero. We iniatialze them with a uniform function
    #   3rd: This is the activation function we want to choose in our hidden layer
    #   4th: This is the number of nodes in the input layer, the number of independent variables.
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu')) 
    # Adding the Output layer
    # We are changing the activation function because we are making a geodemographic segmentation model, we want to have probabilities for the outcome
    #   We will use the sigmoid activation function. It is the heart of the probabilistic approach
    # If you have more than 2 categories for the output layer, then you would use softmax. It is the sigmoid function but applied to a dependent 
    #   variable that has more than two categories
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    # We are applying stochastic gradient descent on the whole ANN
    # Arguments:
    #   1st: This finds the optimal set of weights in the neural network. We need to add an algorithm
    #           This algorithm is the stochastic gradient descent algorithm. There are several types. A very efficient one is called Adam
    #   2nd: This is the loss function within the stochastic gradient descent algorithm. It will be optimized through stochastic 
    #           gradient descent to eventually find the optimal weights
    #           - If the dependent variable has a binary outcome, then this logarithmiic loss function is called binary_crossentropy
    #           - If the dependent variable has more than two outcomes, then this logarithmic loss function is called categorical_crossentropy
    #   3rd: This is just a criterion that you choose to evaluate your model. You typically use the accuracy criterion. It is used to improve the models
    #           performance. It expects a list of metrics but we only use one that is why its in brackets.
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Returning the classifier
    return classifier

# Now, we wrap the whole thing. We create a new classifier that will be the global classifier variable. 
# This new classifier will be built through k-fold cross validation on 10 different training folds
# Arguments:
#   1st: This is the function that built the classifier.
#   2nd: Batch size
#   3rd: epochs    
# We do not input the batch size or the epochs arguments in the keras classifier because those are the arguments we are going to tune. The arguments
#   we are going to tune will be put separately in the gridsearch object we are going to make.
classifier = KerasClassifier(build_fn = build_classifier)

# This is a dictionary of the hyperparameters that we want to optimize, that we want to find the best values.
# You can choose any hyperparameter you want to tune.
parameters = {
        # Hyperparameters
        # These are the values we want to test
        'batch_size': [25, 32],
        'epochs': [100, 500],
        'optimizer': ['adam', 'rmsprop']
        }

# We are now going to implement grid search.
# This is an exhaustive search over specified parameter values for an estimator
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
# We fit grid search to the data set. 
grid_search = grid_search.fit(X_train, y_train)

# This will tell us what the best parameters are
best_parameters = grid_search.best_params_
# This will tell us the best accuracy that result from the best selection of hyperparameters.
best_accuracy = grid_search.best_score_