# Intro to anomaly detection with simple 2D dataset that monitors conditions of computer servers

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
import matplotlib as mpl

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils


def plotDataset(X):
    #  Visualize the example dataset
    pyplot.plot(X[:, 0], X[:, 1], 'bx', mew=2, mec='b', ms=6)
    pyplot.axis([0, 30, 0, 30])
    pyplot.xlabel('Latency (ms)')
    pyplot.ylabel('Throughput (mb/s)')

    # pyplot.show()


# loops
def estimateGaussian(X):
    m, n = X.shape

    means = np.zeros(n)
    variances = np.zeros(n)

    for col in range(n):
        feature = X[:, col]
        means[col] = np.mean(feature)
        variances[col] = np.var(feature)

    return means, variances


# vectorization
def estimateGaussian2(X):
    m, n = X.shape

    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)

    return means, variances

def visualize_contours(X, mu, sigma2):
    # visualize contours of estimated gaussian distribution

    #  Visualize the fit
    utils.visualizeFit(X, mu, sigma2)
    pyplot.xlabel('Latency (ms)')
    pyplot.ylabel('Throughput (mb/s)')
    pyplot.tight_layout()

    pyplot.show()

def selectThreshold(yval, pval):
    epsilons = np.linspace(1.01 * min(pval), max(pval), 1000)

    F1_scores = np.zeros(epsilons.shape[0])

    i = 0
    for epsilon in epsilons:
        # boolean vector that is true (1) for anomaly and false (0) for normal
        predictions = pval < epsilon
        tp = np.sum((yval == 1) & (predictions == 1))
        fp = np.sum((yval == 0) & (predictions == 1))
        fn = np.sum((yval == 1) & (predictions == 0))

        precision = (tp) / (tp + fp)
        recall = (tp) / (tp + fn)

        F1_score = (2 * precision * recall) / (precision + recall)

        F1_scores[i] = F1_score
        i += 1

    bestF1 = np.max(F1_scores)
    bestEpsilon = epsilons[np.argmax(F1_scores)]

    return bestEpsilon, bestF1

#  The following command loads the dataset.
data = loadmat('ex8data1.mat')
X, Xval, yval = data['X'], data['Xval'], data['yval'][:, 0]

#  Estimate mu and sigma2
mu, sigma2 = estimateGaussian2(X)

#  Returns the density of the multivariate normal at each data point (row)
#  of X
p = utils.multivariateGaussian(X, mu, sigma2)

pval = utils.multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval, pval)
print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('Best F1 on Cross Validation Set:  %f' % F1)
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)')

#  Find the outliers in the training set and plot the
outliers = p < epsilon

#  Visualize the fit
utils.visualizeFit(X,  mu, sigma2)
pyplot.xlabel('Latency (ms)')
pyplot.ylabel('Throughput (mb/s)')
pyplot.tight_layout()

#  Draw a red circle around those outliers
pyplot.plot(X[outliers, 0], X[outliers, 1], 'ro', ms=10, mfc='None', mew=2)

pyplot.show()






