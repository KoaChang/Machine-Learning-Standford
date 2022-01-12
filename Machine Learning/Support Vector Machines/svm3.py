# USING DATASET 3

# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Import regular expressions to process emails
import re

import sys

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

import math

# Load from ex6data3
# You will have X, y, Xval, yval as keys in the dict data
data = loadmat('ex6data3.mat')
X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]

# Plot training data
utils.plotData(X, y)

def gaussianKernel(x1,x2,sigma):
    return math.e**(  -1* ( ((np.linalg.norm(x1 - x2))**2)/(2*(sigma)**2) )   )

def dataset3Params(X,y,Xval,yval):
    C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    C_min = 0
    sig_min = 0
    min_error = sys.maxsize

    for c in C:
        for sig in sigma:
            model = utils.svmTrain(X, y, c, gaussianKernel, args=(sig,))
            pred = utils.svmPredict(model,Xval)
            error_val = np.mean(pred != yval)
            if error_val < min_error:
                min_error = error_val
                C_min = c
                sig_min = sig

    return C_min,sig_min


# Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)

# Train the SVM
# model = utils.svmTrain(X, y, C, lambda x1, x2: gaussianKernel(x1, x2, sigma))
model = utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
utils.visualizeBoundary(X, y, model)
print(C, sigma)



