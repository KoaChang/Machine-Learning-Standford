# USING DATASET 1

# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Import regular expressions to process emails
import re

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

import math

# Load from ex6data2
# You will have X, y as keys in the dict data
data = loadmat('ex6data2.mat')
X, y = data['X'], data['y'][:, 0]

# Plot training data
utils.plotData(X, y)

def gaussianKernel(x1,x2,sigma):
    return math.e**(  -1* ( ((np.linalg.norm(x1 - x2))**2)/(2*(sigma)**2) )   )


# SVM Parameters
C = 1
sigma = 0.1

model= utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
utils.visualizeBoundary(X, y, model)


