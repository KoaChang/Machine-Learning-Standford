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

def PCA(X):
    m = X.shape[0]

    sigma = (1/m) * np.matmul(X.T,X)

    U, S, V = np.linalg.svd(sigma)

    return U,S

def projectData(X,U,K):
    Ureduce = U[:,:K]
    projected = np.matmul(X,Ureduce)
    return projected

def recoverData(Z,U,K):
    Ureduce = U[:,:K]
    X_rec = np.matmul(Z, Ureduce.T)
    return X_rec

def PCA2DDataset():
    # Load the dataset into the variable X
    data = loadmat('ex7data1.mat')
    X = data['X']

    # Visualize the example dataset
    pyplot.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=1)
    pyplot.axis([0.5, 6.5, 2, 8])
    pyplot.gca().set_aspect('equal')
    pyplot.grid(False)

    X_norm, mu, sigma = utils.featureNormalize(X)

    #  Run PCA
    U, S = PCA(X_norm)

    #  Draw the eigenvectors centered at mean of data. These lines show the
    #  directions of maximum variations in the dataset.
    fig, ax = pyplot.subplots()
    ax.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=0.25)

    for i in range(2):
        ax.arrow(mu[0], mu[1], 1.5 * S[i] * U[0, i], 1.5 * S[i] * U[1, i],
                 head_width=0.25, head_length=0.2, fc='k', ec='k', lw=2, zorder=1000)

    ax.axis([0.5, 6.5, 2, 8])
    ax.set_aspect('equal')
    ax.grid(False)

    print('Top eigenvector: U[:, 0] = [{:.6f} {:.6f}]'.format(U[0, 0], U[1, 0]))
    print(' (you should expect to see [-0.707107 -0.707107])')

    #  Project the data onto K = 1 dimension
    K = 1
    Z = projectData(X_norm, U, K)
    print('Projection of the first example: {:.6f}'.format(Z[0, 0]))
    print('(this value should be about    : 1.481274)')

    X_rec = recoverData(Z, U, K)
    print('Approximation of the first example: [{:.6f} {:.6f}]'.format(X_rec[0, 0], X_rec[0, 1]))
    print('       (this value should be about  [-1.047419 -1.047419])')

    #  Plot the normalized dataset (returned from featureNormalize)
    fig, ax = pyplot.subplots(figsize=(5, 5))
    ax.plot(X_norm[:, 0], X_norm[:, 1], 'bo', ms=8, mec='b', mew=0.5)
    ax.set_aspect('equal')
    ax.grid(False)
    pyplot.axis([-3, 2.75, -3, 2.75])

    # Draw lines connecting the projected points to the original points
    ax.plot(X_rec[:, 0], X_rec[:, 1], 'ro', mec='r', mew=2, mfc='none')
    for xnorm, xrec in zip(X_norm, X_rec):
        ax.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], '--k', lw=1)

    pyplot.show()

def PCAFace():
    #  Load Face dataset
    data = loadmat('ex7faces.mat')
    X = data['X']

    #  Display the first 100 faces in the dataset
    # utils.displayData(X[:100, :], figsize=(8, 8))

    # Normalize X by subtracting the mean value from each feature
    X_norm, mu, sigma = utils.featureNormalize(X)

    # Run PCA
    U, S = PCA(X_norm)

    # Visualize the top 36 eigenvectors found
    # utils.displayData(U[:, :36].T, figsize=(8, 8))

    #  Project images to the eigen space using the top k eigenvectors
    #  If you are applying a machine learning algorithm
    K = 100
    Z = projectData(X_norm, U, K)

    print('The projected data Z has a shape of: ', Z.shape)

    #  Project images to the eigen space using the top K eigen vectors and
    #  visualize only using those K dimensions
    #  Compare to the original input, which is also displayed
    K = 100
    X_rec = recoverData(Z, U, K)

    # Display normalized data
    utils.displayData(X_norm[:100, :], figsize=(6, 6))
    pyplot.gcf().suptitle('Original faces')

    # display projected data
    utils.displayData(Z[:100, :], figsize=(6, 6))
    pyplot.gcf().suptitle('Projected Data')

    # Display reconstructed data from only k eigenfaces
    utils.displayData(X_rec[:100, :], figsize=(6, 6))
    pyplot.gcf().suptitle('Recovered faces')

def visualizeRGBClusters():
    # this allows to have interactive plot to rotate the 3-D plot
    # The double identical statement is on purpose
    # see: https://stackoverflow.com/questions/43545050/using-matplotlib-notebook-after-matplotlib-inline-in-jupyter-notebook-doesnt
    from matplotlib import pyplot


    A = mpl.image.imread('bird_small.png')
    A /= 255
    X = A.reshape(-1, 3)

    # perform the K-means clustering again here
    K = 16
    max_iters = 10
    initial_centroids = kMeansInitCentroids(X, K)
    centroids, idx = utils.runkMeans(X, initial_centroids,
                                     findClosestCentroids,
                                     computeCentroids, max_iters)

    #  Sample 1000 random indexes (since working with all the data is
    #  too expensive. If you have a fast computer, you may increase this.
    sel = np.random.choice(X.shape[0], size=1000)

    fig = pyplot.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], cmap='rainbow', c=idx[sel], s=8**2)
    ax.set_title('Pixel dataset plotted in 3D.\nColor shows centroid memberships');



    # Subtract the mean to use PCA
    X_norm, mu, sigma = utils.featureNormalize(X)

    # PCA and project the data to 2D
    U, S = PCA(X_norm)
    Z = projectData(X_norm, U, 2)

    fig = pyplot.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.scatter(Z[sel, 0], Z[sel, 1], cmap='rainbow', c=idx[sel], s=64)
    ax.set_title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    ax.grid(False)



