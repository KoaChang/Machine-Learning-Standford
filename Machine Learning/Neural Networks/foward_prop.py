# this code uses forward propagation to hypothesize the handwritten digit based on already learned
# parameters theta that was given to us in assignment

import numpy as np
import math
from matplotlib import pyplot
from scipy import optimize
import random

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat
import utilsWk4

def sigmoid(z):
    if type(z) == np.ndarray:
        z = z.astype('float64')
        if z.ndim == 1:
            for i in range(len(z)):
                value = z[i]
                z[i] = 1 / (1 + math.e ** (-1*value))
            return z
        else:
            z = z.astype('float64')
            for row in range(z.shape[0]):
                for col in range(z.shape[1]):
                    value = z[row][col]
                    value = 1 / (1 + math.e ** (-1*value))
                    z[row][col] = value
            return z
    else:
        return 1 / (1 + math.e ** (-1*z))

#  training data stored in arrays X, y
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in
# MATLAB where there is no index 0
y[y == 10] = 0

m,n = X.shape

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# Load the .mat file, which returns a dictionary
weights = loadmat('ex3weights.mat')

# get the model weights from the dictionary
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing,
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

def predict(Theta1, Theta2, X):
    m,n = X.shape
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    a2 = sigmoid(np.matmul(X,Theta1.T))
    a2 = np.concatenate([np.ones((m, 1)), a2], axis=1)
    a3 = np.matmul(a2,Theta2.T)
    hypothesis = sigmoid(a3)

    predictions = []
    for row in hypothesis:
        number = np.argmax(row,axis=0)
        predictions.append(number)

    return predictions

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))


# Running the following cell will display a single image and the neural network's prediction. You can run the code
# multiple times to see predictions for different images.
indices = np.random.permutation(m)

if indices.size > 0:
    i, indices = indices[0], indices[1:]
    utilsWk4.displayData(X[i, :], figsize=(4, 4))
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')



