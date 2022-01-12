import numpy as np
import math
from matplotlib import pyplot
from scipy import optimize
import random

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat
import utilsWk5

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

def sigmoid_gradient(z):
    if type(z) == np.ndarray:
        z = z.astype('float64')
        for row in range(z.shape[0]):
            for col in range(z.shape[1]):
                value = z[row][col]
                value = sigmoid(value) * (1-sigmoid(value))
                z[row][col] = value
    else:
        return sigmoid(z) * (1-sigmoid(z))
    return z

def randInitializeWeights(L_in,L_out,epsilon_init=0.12):
    # You need to return the following variables correctly
    W = np.zeros((L_out, 1 + L_in))

    # ====================== YOUR CODE HERE ======================

    # Randomly initialize the weights to small values
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    # ============================================================
    return W

# Theta1 has size 25 x 401
# Theta2 has size 10 x 26

# training data stored in arrays X, y
data = loadmat('ex4data1.mat')
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in
# MATLAB where there is no index 0
y[y == 10] = 0


# Number of training examples
m = y.size

# Randomly select 100 data points to display
# rand_indices = np.random.choice(m, 100, replace=False)
# sel = X[rand_indices, :]
#
# utils.displayData(sel)

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# Load the weights into variables Theta1 and Theta2
weights = loadmat('ex4weights.mat')

# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing,
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

# Unroll parameters
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

lambda_ = 1

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)


# cost function without regularization
def costFunction(theta_unrolled,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y,lambda_):
    # Reshape theta_unrolled back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(theta_unrolled[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1))) # 25 x 401

    Theta2 = np.reshape(theta_unrolled[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1))) # 10 x 26

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    m,n = X.shape

    # z values is the actual matrix made by the multiplication of input units and theta, then activations is simply
    # passing those z values into the sigmoid function
    # find hypothesis
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    a1 = X    # 5000 * 401
    a2 = sigmoid(np.matmul(X,Theta1.T))     # 5000 * 26
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    a3 = sigmoid(np.matmul(a2,Theta2.T))
    hypothesis = a3
    # hypothesis shape 5000*10

    # Encode the labels as vectors containing only values 0 or 1
    y_matrix = y.reshape(-1)
    y_matrix = np.eye(num_labels)[y_matrix]
    # y matrix shape 5000*10

    ones = np.ones(y_matrix.shape)

    # Compute cost without regularization
    cost = (1/m) * np.sum(np.multiply(-1*y_matrix,np.log(hypothesis)) - np.multiply((ones-y_matrix),np.log(ones-hypothesis)) )

    # Compute cost J with regularization terms
    # Add regularization terms excluding the Theta columns for the bias units
    reg = (lambda_/(2*m)) * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))

    cost = cost+reg

    # could implement backprop with a for loop going through each training example, forward propagating it,
    # then backpropagating it to calculate each delta value and updating the grafdient matrix for each theta.

    # or can straight up use vectorized approach which is much easier, simpler, and straight forward as you
    # can calculate each delta in one go and accumulate it all at once instead of looping throughout and adding to each
    # theta gradient through every forward prop and back prop for each training example.

    # d3 shape 5000 * 10
    d3 = hypothesis - y_matrix

    # Excluding the first column of Theta2 is because the hidden layer bias unit
    # has no connection to the input layer, so we do not use backpropagation for it
    d2 = np.matmul(d3,Theta2[:,1:]) * sigmoid_gradient(np.matmul(X,Theta1.T))

    # 10 * 26
    delta_2 = np.matmul(d3.T,a2)

    # 25 * 401
    delta_1 = np.matmul(d2.T,a1)

    Theta1_grad = (1/m) * delta_1
    Theta2_grad = (1/m) * delta_2

    # regularize the gradients for each theta based on lambda. Don't regularize bias units which is always first column
    # in theta matrix

    Theta1_grad[:,1:] += (lambda_/m) * Theta1[:,1:]
    Theta2_grad[:,1:] += (lambda_ /m) * Theta2[:,1:]

    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return cost, grad

# print(costFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda_))

def optimize_algorithm(initial_nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y,lambda_):
    #  After you have completed the assignment, change the maxiter to a larger
    #  value to see how more training helps.
    options = {'maxiter': 100}

    #  You should also try different values of lambda
    lambda_ = 1

    # Create "short hand" for the cost function to be minimized
    costFunc = lambda p: costFunction(p, input_layer_size,
                                            hidden_layer_size,
                                            num_labels, X, y, lambda_)

    # Now, costFunction is a function that takes in only one argument
    # (the neural network parameters)
    res = optimize.minimize(costFunc,
                            initial_nn_params,
                            jac=True,
                            method='TNC',
                            options=options)

    # get the solution of the optimization
    nn_params = res.x

    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    return (Theta1,Theta2)

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

Theta1, Theta2 = optimize_algorithm(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda_)

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))



