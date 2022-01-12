import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import optimize
from utils import plotDecisionBoundary

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

def plotData(X,Y):
    # Find Indices of Positive and Negative Examples
    pos = Y == 1
    neg = Y == 0

    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y=1','y=0'])
    plt.show()


def mapFeature(X1, X2):
    """
    Maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.

    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.

        Inputs X1, X2 must be the same size.

    degree: int, optional
        The polynomial degree.

    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """
    degree = 6
    out = np.ones(X.shape[0])[:, np.newaxis]
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            out = np.hstack((out, np.multiply(np.power(X1, i - j), np.power(X2, j))[:, np.newaxis]))
    return out

def costFunction(theta,X,Y,lambda_):
    Y = np.expand_dims(Y, axis=1)
    theta = np.expand_dims(theta, axis=1)

    m = X.shape[0]
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

def optimized_algorithm(theta,X,Y,lambda_):
    res = optimize.minimize(costFunction,
                            theta,
                            (X, Y, lambda_),
                            jac=True,
                            method='TNC',
                            options={'maxiter': 3000})

    # The fun property of OptimizeResult object returns the value of costFunction at optimized theta
    cost = res.fun

    # The optimized theta is in the x property of the result
    theta = res.x

    return theta

def plotDecisionBoundary(theta, X, y):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with + for the positive examples and o for  the negative examples.

    Parameters
    ----------

    theta : array_like
        Parameters for logistic regression. A vector of shape (n+1, ).

    X : array_like
        The input dataset. X is assumed to be  a either:
            1) Mx3 matrix, where the first column is an all ones column for the intercept.
            2) MxN, N>3 matrix, where the first column is all ones.

    y : array_like
        Vector of data labels of shape (m, 1).
    """
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))

    def mapFeaturePlot(X1, X2):
        degree = 6
        out = np.ones(1)
        for i in range(1, degree + 1):
            for j in range(0, i + 1):
                out = np.hstack((out, np.multiply(np.power(X1, i - j), np.power(X2, j))))
        return out

    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = np.dot(mapFeaturePlot(u[i], v[j]), theta)

    pos = y.flatten() == 1
    neg = y.flatten() == 0
    X = data[:, 0:2]

    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)

    plt.contour(u, v, z, 0)
    plt.title('Figure 4: Training data with decision boundary (λ = 1)\n', fontsize=14)
    plt.xlabel('Microchip Test1')
    plt.ylabel('Microchip Test2')
    plt.legend(['y = 1', 'y = 0'], loc='upper right')
    plt.show()

def predict(theta,X):
    theta = np.expand_dims(theta, axis=1)
    hypothesis = sigmoid(np.matmul(X,theta))

    pred = []
    for i in range(len(hypothesis)):
        if hypothesis[i][0] >= 0.5:
            pred.append(1)
        else:
            pred.append(0)

    return pred


data = np.genfromtxt('ex2data2.txt',delimiter=',',dtype=float)
X = data[:,0:2]
Y = data[:,2]
# Note that mapFeature also adds a column of ones for us, so the intercept term is included
X = mapFeature(X[:, 0], X[:, 1])

# Setup the data matrix appropriately
m, n = X.shape
# Convert label (y) from 1D array to 2D array of shape (m, 1)

initial_theta = np.zeros(n)
lambda_= 1

cost,grad = costFunction(initial_theta,X,Y,lambda_)

theta = optimized_algorithm(initial_theta,X,Y,lambda_)

predictions = predict(theta,X)
print('Percent accuracy of our model: {}'.format(np.mean(predictions == Y)*100))

plotDecisionBoundary(theta, X, Y)
