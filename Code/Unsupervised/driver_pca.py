# driver_pca.py

import load_mnist
import numpy as np
import pca

# Things to try:
# Change number of training data samples: ntrain up to 60000
# Change variance capture: variance capture (greater than 0 and less equal to 1)
ntrain = 6000
variance_capture = 0.99
# load mnist data set
X,_ = load_mnist.load_mnist(ntrain)
# subtract mean
X = X - np.mean(X,axis=1,keepdims=True)
# perform pca
model = pca.pca(variance_capture)
reduced_X = model.compute_reduced(X)