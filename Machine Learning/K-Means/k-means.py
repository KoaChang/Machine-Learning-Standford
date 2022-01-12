# Scientific and vector computation for python
import numpy as np

# Import regular expressions to process emails
import re

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

try:
    pyplot.rcParams["animation.html"] = "jshtml"
except ValueError:
    pyplot.rcParams["animation.html"] = "html5"

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils
import sys

# returns index of closest centroid to particular training example
def minimize(training_example,centroids):
    closest_centroid = 0
    min_distance = sys.maxsize
    for i in range(centroids.shape[0]):
        if np.linalg.norm(training_example - centroids[i])**2 < min_distance:
            min_distance = np.linalg.norm(training_example - centroids[i])**2
            closest_centroid = i
    return closest_centroid

def findClosestCentroids(X,centroids):
    m = X.shape[0]

    idx = np.zeros(m,dtype=int)
    for i in range(m):
        closest_centroid = minimize(X[i],centroids)
        idx[i] = closest_centroid

    return idx

# vectorized implementation much simpler
def findClosestCentroids2(X,centroids):
    m = X.shape[0]

    idx = np.zeros(m,dtype=int)

    for i in range(m):
        training_example = X[i]
        distances = np.sum((centroids - training_example)**2, axis=1)
        idx[i] = np.argmin(distances)

    return idx

def computeCentroids(X,idx,K):

    centroids = np.zeros((K,X.shape[1]))

    for i in range(K):
        training_examples = tuple([idx == i])
        centroids[i] = np.sum( X[training_examples],axis=0 )/np.sum(training_examples)

    return centroids

def example_dataset():
    # Load an example dataset
    data = loadmat('ex7data2.mat')

    # Settings for running K-Means
    X = data['X']
    K = 3
    max_iters = 10

    # For consistency, here we set centroids to specific values
    # but in practice you want to generate them automatically, such as by
    # settings them to be random examples (as can be seen in
    # kMeansInitCentroids).
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])


    # Run K-Means algorithm. The 'true' at the end tells our function to plot
    # the progress of K-Means
    centroids, idx, anim = utils.runkMeans(X, initial_centroids,
                                           findClosestCentroids2, computeCentroids, max_iters, True)

    pyplot.show()


def kMeansInitCentroids(X, K):
    """
    This function initializes K centroids that are to be used in K-means on the dataset x.

    Parameters
    ----------
    X : array_like
        The dataset of size (m x n).

    K : int
        The number of clusters.

    Returns
    -------
    centroids : array_like
        Centroids of the clusters. This is a matrix of size (K x n).

    Instructions
    ------------
    You should set centroids to randomly chosen examples from the dataset X.
    """
    m, n = X.shape

    # You should return this values correctly
    centroids = np.zeros((K, n))

    # ====================== YOUR CODE HERE ======================

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]

    # =============================================================
    return centroids

def KMeansImageCompression():
    # ======= Experiment with these parameters ================
    # You should try different values for those parameters
    K = 16
    max_iters = 10

    # Load an image of a bird 128 x 128 pixels
    # Change the file name and path to experiment with your own images
    A = mpl.image.imread('bird_small.png')
    # ==========================================================

    # Divide by 255 so that all values are in the range 0 - 1
    A /= 255

    # Reshape the image into an Nx3 matrix where N = number of pixels.
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X that we will use K-Means on.
    X = A.reshape(-1, 3)

    # When using K-Means, it is important to randomly initialize centroids
    # You should complete the code in kMeansInitCentroids above before proceeding
    initial_centroids = kMeansInitCentroids(X, K)

    # Run K-Means
    centroids, idx = utils.runkMeans(X, initial_centroids,
                                     findClosestCentroids,
                                     computeCentroids,
                                     max_iters)

    # We can now recover the image from the indices (idx) by mapping each pixel
    # (specified by its index in idx) to the centroid value
    # Reshape the recovered image into proper dimensions
    X_recovered = centroids[idx, :].reshape(A.shape)

    # Display the original image, rescale back by 255
    fig, ax = pyplot.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(A*255)
    ax[0].set_title('Original')
    ax[0].grid(False)

    # Display compressed image, rescale back by 255
    ax[1].imshow(X_recovered*255)
    ax[1].set_title('Compressed, with %d colors' % K)
    ax[1].grid(False)

    pyplot.show()


KMeansImageCompression()
