import numpy as np
import math
from matplotlib import pyplot
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat
import utils

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

def costFunction(theta,X,Y,lambda_):

    # this if statement is to handle the weird glitch in the optimize function that passes theta as shape
    # (1,401) for some iterations on accident for no reason
    Y = np.expand_dims(Y, axis=1)
    if theta.shape == (1,401):
        theta = theta.T
    else:
        theta = np.expand_dims(theta, axis=1)

    m = X.shape[0]
    n = X.shape[1]
    ones = np.ones((m, 1))
    gradient = np.zeros(n)

    # calculate hypothesis
    hypothesis = sigmoid(np.matmul(X, theta))

    # calculate cost. This is the average of all the costs of the hypothesized classification
    # by our function parameterized by theta from the actual y-values. Don't count the of theta0 in regularization
    # part as theta0 is excluded in regularization.
    cost = (1 / m) * (np.matmul(-1 * Y.T, np.log(hypothesis)) - np.matmul((ones - Y).T, np.log(ones - hypothesis))) + (lambda_/(2*m)) * np.matmul(theta[1:,:].T,theta[1:,:])

    # calculate gradient. Cannot quite use vectorization here as we could in cost because matrices won't line up.
    gradient[0] = (1/m) * (np.matmul(X[:,0].T,hypothesis-Y))

    for i in range(1,n):
        gradient[i] = (1 / m) * np.matmul(X[:,i].T, hypothesis - Y) + (lambda_/m) * theta[i]

    return (cost, gradient)

#
# # Randomly select 100 data points to display
# rand_indices = np.random.choice(m, 100, replace=False)
# sel = X[rand_indices, :]
#
# utils.displayData(sel)


# # test values for the parameters theta
# theta_t = np.array([-2, -1, 1, 2], dtype=float)
#
# # test values for the inputs
# X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
#
# # test values for the labels
# y_t = np.array([1, 0, 1, 0, 1])
#
# # test value for the regularization parameter
# lambda_t = 3
#
# J, grad = costFunction(theta_t, X_t, y_t, lambda_t)
#
# print('Cost         : {}'.format(J))
# print('Expected cost: 2.534819')
# print('-----------------------')
# print('Gradients:')
# print(' [{}, {}, {}, {}]'.format(*grad))
# print('Expected gradients:')
# print(' [0.146561, -0.548558, 0.724722, 1.398003]')

def optimized_algorithm(theta,X,Y,lambda_):
    options = {'maxiter': 50}
    res = optimize.minimize(costFunction,
                            theta,
                            (X, Y, lambda_),
                            jac=True,
                            method='CG',
                            options=options)

    # The fun property of OptimizeResult object returns the value of costFunction at optimized theta
    cost = res.fun

    # The optimized theta is in the x property of the result
    theta = res.x

    return theta

# trains each logistic regression classifier
def oneVSAll(X,y,num_labels,lambda_):
    for i in range(num_labels):
        copy_y = np.copy(y)
        true = (copy_y == i)
        false = (copy_y != i)
        copy_y[false] = 0
        copy_y[true] = 1

        theta = np.zeros(n + 1)
        theta = optimized_algorithm(theta, X, copy_y, lambda_)
        all_theta[i] = theta

    return all_theta

def predictOneVsAll(all_theta,X):
    probabilities = np.matmul(X,all_theta.T)
    probabilities = sigmoid(probabilities)

    predictions = []

    for row in probabilities:
        number = np.argmax(row,axis=0)
        predictions.append(number)

    return predictions

# 20x20 Input Images of Digits
input_layer_size  = 400

# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10

#  training data stored in arrays X, y
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in
# MATLAB where there is no index 0
y[y == 10] = 0

m,n = X.shape

# 10 classifications for each different number from 0-9
num_labels = 10

# add one more feature to each theta classification row which will represent the constant (y-intercept) for the parameter
all_theta = np.zeros((num_labels, n + 1))

# add column of ones to X
X = np.concatenate([np.ones((m, 1)), X], axis=1)

lambda_ = 0.1

all_theta = oneVSAll(X,y,num_labels,lambda_)

# get accuracy of predictions
predictions = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(predictions == y) * 100))






