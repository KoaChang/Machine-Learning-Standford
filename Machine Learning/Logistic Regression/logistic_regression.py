import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import optimize
from utils import plotDecisionBoundary


# The log function used in the cost function for logistic regression is ln with base e and
# NOT the common logarithm which is base 10.

def plotData(X,Y):
    # Find Indices of Positive and Negative Examples
    pos = Y == 1
    neg = Y == 0

    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(['admitted','not admitted'])

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

# must use flattened arrays for theta and Y because optimization algorithm in scipy only
# accepts those arguments to have no column dimension at all. 1D only.

def costFunction(theta,X,Y,optimized=True):
# returns a tuple, first item is the scalar cost value, and the second is a column vector of
# size n which represents the gradient for each parameter theta.

    # expand dimensions to proper matrices/vectors with an actual column value of 1 instead of None.
    # this allows us to do our vectorization calculations.

    if optimized:
        Y = np.expand_dims(Y, axis=1)
        theta = np.expand_dims(theta, axis=1)

    m = X.shape[0]
    ones = np.ones((m,1))

    # calculate hypothesis
    hypothesis = sigmoid(np.matmul(X,theta))

    # calculate cost. This is the average of all the costs of the hypothesized classification
    # by our function parameterized by theta from the actual y-values.
    cost = (1/m) * (   np.matmul(-1*Y.T,np.log(hypothesis)) - np.matmul((ones-Y).T, np.log(ones-hypothesis))   )

    # calculate gradient
    gradient = (1/m) * np.matmul(X.T,hypothesis-Y)

    return (cost,gradient)


def optimized_algorithm(initial_theta,X,Y):
    # set options for optimize.minimize
    options = {'maxiter': 400}

    # see documention for scipy's optimize.minimize  for description about
    # the different parameters
    # The function returns an object `OptimizeResult`
    # We use truncated Newton algorithm for optimization which is
    # equivalent to MATLAB's fminunc
    # See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
    res = optimize.minimize(costFunction,
                            initial_theta,
                            (X, Y),
                            jac=True,
                            method='TNC',
                            options=options)

    # the fun property of `OptimizeResult` object returns
    # the value of costFunction at optimized theta
    cost = res.fun

    # the optimized theta is in the x property
    theta = res.x

    # print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
    # print('Expected cost (approx): 0.203\n');
    #
    # print('theta:')
    # print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
    # print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

    return theta

def gradient_descent(learning_rate,iterations):
    data = np.genfromtxt('ex2data1.txt', delimiter=',', dtype=float)
    X = data[:, 0:2]
    X = np.concatenate((np.ones((X.shape[0], 1), dtype=int), X), axis=1)
    Y = data[:, 2:3]
    m, n = X.shape[0], X.shape[1]
    theta = np.zeros((n,1))

    for i in range(iterations):
        theta = theta - learning_rate*(costFunction(theta,X,Y,optimized=False)[1])

    return theta
# returns the optimized parameters of theta for logistic regression that fits our training
# data the best with the least cost. Coded on my own with choosing my own learning rate
# alpha and making sure the cost is decreasing over every iteration. (might be helpful to plot
# a graph of iterations vs cost of theta to ensure our learning algorithm for the data is improving).
# Since this is our own implementation of gradient descent, it will not be as good and as fast
# as more optimal and complex algorithms used in libraries like scipy.optimize or fminunc.

# not as good as optimized super gradient descent algorithm in python library that works everytime.
# If we use gradient descent on our own we must use trial and error to identify best learning
# rate and iterations.

def plotDecisionBoundry():
    data = np.genfromtxt('ex2data1.txt',delimiter=',',dtype=float)
    X = data[:,0:2]
    X = np.concatenate((np.ones((X.shape[0],1),dtype=int),X),axis=1)
    Y = data[:,2]
    m,n = X.shape[0], X.shape[1]
    initial_theta = np.zeros(n)

    theta = optimized_algorithm(initial_theta,X,Y)

    plotDecisionBoundary(plotData, theta, X, Y)
    plt.show()

def predict(theta,X):
    m = X.shape[0]
    theta = np.expand_dims(theta, axis=1)
    predictions = sigmoid(np.matmul(X,theta))
    pred = []
    for i in range(m):
        if predictions[i][0] >= 0.5:
            pred.append(1)
        else:
            pred.append(0)

    return pred

data = np.genfromtxt('ex2data1.txt',delimiter=',',dtype=float)
X = data[:,0:2]
X = np.concatenate((np.ones((X.shape[0],1),dtype=int),X),axis=1)
Y = data[:,2]

initial_theta = np.zeros(X.shape[1])
theta = optimized_algorithm(initial_theta,X,Y)

predictions = predict(theta,X)

print('Percent accuracy of our model: {}'.format(np.mean(predictions==Y)))

