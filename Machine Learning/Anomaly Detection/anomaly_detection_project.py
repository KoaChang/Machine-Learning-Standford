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

# loops
def estimateGaussian(X):
    m,n = X.shape

    means = np.zeros(n)
    variances = np.zeros(n)

    for col in range(n):
        feature = X[:,col]
        means[col] = np.mean(feature)
        variances[col] = np.var(feature)

    return means,variances

# vectorization
def estimateGaussian2(X):
    m,n = X.shape

    means = np.mean(X,axis=1)
    variances = np.var(X,axis=1)

    return means,variances

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

#  Loads the second dataset. You should now have the
#  variables X, Xval, yval in your environment
data = loadmat('ex8data2.mat')
X, Xval, yval = data['X'], data['Xval'], data['yval'][:, 0]

# Apply the same steps to the larger dataset
mu, sigma2 = estimateGaussian(X)

#  Training set
p = utils.multivariateGaussian(X, mu, sigma2)

#  Cross-validation set
pval = utils.multivariateGaussian(Xval, mu, sigma2)

#  Find the best threshold
epsilon, F1 = selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: %.2e' % epsilon)
print('Best F1 on Cross Validation Set          : %f\n' % F1)
print('  (you should see a value epsilon of about 1.38e-18)')
print('   (you should see a Best F1 value of      0.615385)')
print('\n# Outliers found: %d' % np.sum(p < epsilon))


