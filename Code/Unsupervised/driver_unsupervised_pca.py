# driver_unsupervised_pca.py

import load_mnist
import matplotlib.pyplot as plt
import numpy as np
import pca
import plot_data

# Things to try:
# Change number of training data samples: ntrain up to 10000
# Change variance capture: variance capture (greater than 0 and less equal to 1)
ntrain = 6000
variance_capture = 0.99
# load mnist data set
X,Y = load_mnist.load_mnist(ntrain)
plot_data.plot_data_mnist(X,Y)
# compute mean
Xmean = np.mean(X,axis=1,keepdims=True)
# perform pca
model = pca.pca(variance_capture)
reduced_X = model.compute_reduced(X-Xmean)
# plot data after reduction of dimension (add back Xmean)
plot_data.plot_data_mnist(reduced_X+Xmean,Y)
plt.show()