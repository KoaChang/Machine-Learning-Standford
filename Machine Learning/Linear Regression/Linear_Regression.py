import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# normal equation
# data = np.genfromtxt('ex1data2.txt',delimiter=',',dtype=int)
# X = data[:,0:2]
# Y = data[:,2]
#
# X = np.concatenate((np.ones((47,1),dtype=int),X),axis=1)
#
# theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),Y)
# print(theta)


# compute cost
def computecost(X,theta,Y):
    m = X.shape[0]
    cost = (1/2*m) * np.sum((np.matmul(X,theta) - Y)**2)
    return cost

# gradient descent
def feature_normalize(X):
    for i in range(X.shape[1]):
        column = X[:,i]
        std = np.std(column)
        mean = np.std(column)
        for idx,item in enumerate(column):
            column[idx] = (item - mean)/std
        column = np.expand_dims(column, axis=1)
    return X

def gradientDescent(data):
    X = data[:,0:2]
    Y = data[:,2:3]

    X = feature_normalize(X)

    X = np.concatenate((np.ones((X.shape[0],1),dtype=int),X),axis=1)
    iterations = 1500
    learning_rate = 0.01
    m = X.shape[0]

    # initialize all initers that minimize the cost function J which is the square differences between predicted values and real values
    theta = np.array([[0],[0],[0]])

    total_costs = []

    for i in range(iterations):
        total_costs.append(computecost(X, theta, Y))
        theta = theta - (learning_rate / m) * (np.matmul(X.T, np.matmul(X, theta) - Y))

    # plot the cost as a function of the number of iterations to show how as gradient descent continues its iterations,
    # the cost continuously goes down until theta reaches its optimum value that minimizes cost.
    # plt.plot(np.arange(0,iterations),total_costs)
    # plt.show()

    return theta

data = np.genfromtxt('ex1data2.txt', delimiter=',')
print(gradientDescent(data))


# writing matrix to file
# mat = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# df = pd.DataFrame(data=mat.astype(float))
# df.to_csv('outfile.csv', sep=' ', header=False, float_format='%.2f', index=False)

# loading matrix from file
# A = np.loadtxt('outfile.csv')
# A = A.astype(int)
# print(A)