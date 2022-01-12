# The purpose of this exercise, is to implement regularized linear regression and use it to study models with
# different bias-variance properties.

# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils


# Load from ex5data1.mat, where all variables will be store in a dictionary
data = loadmat('ex5data1.mat')

# Extract train, test, validation data from dictionary
# and also convert y's form 2-D matrix (MATLAB format) to a numpy vector
X, y = data['X'], data['y'][:, 0]
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]

# cross validation sets
Xval, yval = data['Xval'], data['yval'][:, 0]

# m = Number of training examples
m = y.size

# Plot training data
# pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1)
# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)');
# pyplot.show()

def linearRegCostFunction(X,y,theta,lambda_):
    if y.ndim == 1:
        y = np.expand_dims(y, axis=1)

    if theta.ndim == 1:
        theta = np.expand_dims(theta, axis=1)

    hypothesis = np.matmul(X,theta)

    m,n = X.shape

    cost = (1/(2*m)) * np.sum((hypothesis - y)**2) + (lambda_/(2*m)) * np.sum(theta[1:,:]**2)

    gradient = np.zeros(theta.shape[0])

    gradient[0] = (1/m) * np.matmul((hypothesis - y).T,X[:,0:1])
    for i in range(1,n):
        gradient[i] = (1/m) * np.matmul((hypothesis - y).T,X[:,i:i+1]) + (lambda_/m) * theta[i][0]

    return (cost,gradient)

def plot_fit(X,y,lambda_):
    # add a columns of ones for the y-intercept
    X_ones = np.concatenate([np.ones((m, 1)), X], axis=1)

    theta = utils.trainLinearReg(linearRegCostFunction, X_ones, y, lambda_)

    #  Plot fit over the data
    pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1.5)
    pyplot.xlabel('Change in water level (x)')
    pyplot.ylabel('Water flowing out of the dam (y)')
    pyplot.plot(X_ones, np.dot(X_ones, theta), '--', lw=2);
    pyplot.show()

def learningCurve(X, y, Xval, yval, lambda_=0):
    m = X.shape[0]
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(1,m+1):
        theta = utils.trainLinearReg(linearRegCostFunction,X[:i,:],y[:i],lambda_)
        error_train[i-1], _ = linearRegCostFunction(X[:i,:],y[:i],theta,0)
        error_val[i-1], _ = linearRegCostFunction(Xval,yval,theta,0)

    return error_train, error_val

def plot_learning_curve(X,y,Xval,Yval):
    m = X.shape[0]

    X_ones = np.concatenate([np.ones((m, 1)), X], axis=1)
    Xval_ones = np.concatenate([np.ones((yval.size, 1)), Xval], axis=1)
    error_train, error_val = learningCurve(X_ones, y, Xval_ones, yval, lambda_=0)

    pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
    pyplot.title('Learning curve for linear regression')
    pyplot.legend(['Train', 'Cross Validation'])
    pyplot.xlabel('Number of training examples')
    pyplot.ylabel('Error')
    pyplot.axis([0, 13, 0, 150])

    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('\t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

    pyplot.show()

# plot_learning_curve(X,y,Xval,yval)

def polyFeatures(X, p):
    X_poly = np.zeros((X.shape[0],p))
    for i in range(0,p+1):
        X_poly[:,i:i+1] = X**(i+1)

    return X_poly

# apply polynomial features to training, validation, and test sets
p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = utils.featureNormalize(X_poly)
X_poly = np.concatenate([np.ones((m, 1)), X_poly], axis=1)

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.concatenate([np.ones((ytest.size, 1)), X_poly_test], axis=1)

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.concatenate([np.ones((yval.size, 1)), X_poly_val], axis=1)

# plot the fit of data and learning curve when lambda is 0
def plotlambda0():
    lambda_ = 0
    theta = utils.trainLinearReg(linearRegCostFunction, X_poly, y,
                                 lambda_=lambda_, maxiter=55)

    # Plot training data and fit
    pyplot.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')

    utils.plotFit(polyFeatures, np.min(X), np.max(X), mu, sigma, theta, p)

    pyplot.xlabel('Change in water level (x)')
    pyplot.ylabel('Water flowing out of the dam (y)')
    pyplot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
    pyplot.ylim([-20, 50])

    pyplot.figure()
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
    pyplot.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val)

    pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
    pyplot.xlabel('Number of training examples')
    pyplot.ylabel('Error')
    pyplot.axis([0, 13, 0, 100])
    pyplot.legend(['Train', 'Cross Validation'])

    print('Polynomial Regression (lambda = %f)\n' % lambda_)
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

    pyplot.show()

def plotlambda1():
    lambda_ = 1
    theta = utils.trainLinearReg(linearRegCostFunction, X_poly, y,
                                 lambda_=lambda_, maxiter=55)

    # Plot training data and fit
    pyplot.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')

    utils.plotFit(polyFeatures, np.min(X), np.max(X), mu, sigma, theta, p)

    pyplot.xlabel('Change in water level (x)')
    pyplot.ylabel('Water flowing out of the dam (y)')
    pyplot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
    pyplot.ylim([-20, 50])

    pyplot.figure()
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
    pyplot.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val)

    pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
    pyplot.xlabel('Number of training examples')
    pyplot.ylabel('Error')
    pyplot.axis([0, 13, 0, 100])
    pyplot.legend(['Train', 'Cross Validation'])

    print('Polynomial Regression (lambda = %f)\n' % lambda_)
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

    pyplot.show()

def plotlambda100():
    lambda_ = 100
    theta = utils.trainLinearReg(linearRegCostFunction, X_poly, y,
                                 lambda_=lambda_, maxiter=55)

    # Plot training data and fit
    pyplot.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')

    utils.plotFit(polyFeatures, np.min(X), np.max(X), mu, sigma, theta, p)

    pyplot.xlabel('Change in water level (x)')
    pyplot.ylabel('Water flowing out of the dam (y)')
    pyplot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
    pyplot.ylim([-20, 50])

    pyplot.figure()
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
    pyplot.plot(np.arange(1, 1 + m), error_train, np.arange(1, 1 + m), error_val)

    pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
    pyplot.xlabel('Number of training examples')
    pyplot.ylabel('Error')
    pyplot.axis([0, 13, 0, 100])
    pyplot.legend(['Train', 'Cross Validation'])

    print('Polynomial Regression (lambda = %f)\n' % lambda_)
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f' % (i + 1, error_train[i], error_val[i]))

    pyplot.show()

def validationCurve(X, y, Xval, yval):
    lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    error_train = np.zeros(len(lambdas))
    error_val = np.zeros(len(lambdas))

    for idx, lambda_ in enumerate(lambdas):
        theta = utils.trainLinearReg(linearRegCostFunction,X,y,lambda_)
        error_train[idx],_ = linearRegCostFunction(X,y,theta,0)
        error_val[idx],_ = linearRegCostFunction(Xval,yval,theta,0)

    return lambdas, error_train, error_val

def plotValidationCurve():
    lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

    pyplot.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
    pyplot.legend(['Train', 'Cross Validation'])
    pyplot.xlabel('lambda')
    pyplot.ylabel('Error')

    print('lambda\t\tTrain Error\tValidation Error')
    for i in range(len(lambda_vec)):
        print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))

    pyplot.show()

def testCurve(X,y,Xtest,ytest,lambda_):
    m = X.shape[0]

    error_train = np.zeros(m)
    error_test = np.zeros(m)

    for i in range(1,m+1):
        theta = utils.trainLinearReg(linearRegCostFunction,X[:i],y[:i],lambda_)
        error_train[i - 1], _ = linearRegCostFunction(X[:i],y[:i], theta, lambda_=0)
        error_test[i-1],_ = linearRegCostFunction(Xtest,ytest,theta,lambda_=0)

    return error_train,error_test

# plot the test curve with best lambda found which was 3 found from earlier.
def plotTestCurve():
    lambda_ = 3
    theta = utils.trainLinearReg(linearRegCostFunction, X_poly, y,
                                 lambda_=lambda_, maxiter=55)

    # Plot training data and fit
    pyplot.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')

    utils.plotFit(polyFeatures, np.min(X), np.max(X), mu, sigma, theta, p)

    pyplot.xlabel('Change in water level (x)')
    pyplot.ylabel('Water flowing out of the dam (y)')
    pyplot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
    pyplot.ylim([-20, 50])

    pyplot.figure()
    error_train, error_test = testCurve(X_poly, y, X_poly_test, ytest, lambda_)
    pyplot.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_test)

    pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
    pyplot.xlabel('Number of training examples')
    pyplot.ylabel('Error')
    pyplot.axis([0, 13, 0, 100])
    pyplot.legend(['Train', 'Test'])

    print('Polynomial Regression (lambda = %f)\n' % lambda_)
    print('# Training Examples\tTrain Error\tTest Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_test[i]))

    pyplot.show()

# learning curves from randomly selected sets
def randomLearningCurve(X,y,Xtest,ytest,lambda_):
    # Choose i examples randomly from training and validation set to make learning curve.
    # For each i examples you choose to plot on learning curve, do 50 iterations to ensure you get a lot of random
    # sets and take the average error of them all.

    m = X.shape[0]

    error_train = np.zeros(m)
    error_test = np.zeros(m)

    for i in range(1,13):
        total_train_error = 0
        total_val_error = 0
        for _ in range(50):
            random_indices = np.random.choice(m,size = i)
            X_train = X[random_indices]
            y_train = y[random_indices]
            X_test = Xtest[random_indices]
            y_test = ytest[random_indices]

            theta = utils.trainLinearReg(linearRegCostFunction, X_train, y_train, lambda_)
            total_train_error += linearRegCostFunction(X_train, y_train, theta, lambda_=0)[1]
            total_val_error += linearRegCostFunction(X_test, y_test, theta, lambda_=0)[1]

        error_train[i-1] = total_train_error/50
        error_test[i-1] = total_val_error/50

    return error_train, error_test