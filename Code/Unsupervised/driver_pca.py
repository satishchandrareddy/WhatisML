# pca.py

import load_mnist
import numpy as np
import pca


# load mnist data set
X,_,_,_ = load_mnist.load_mnist(6000,1000)
# subtract mean
X = X - np.mean(X,axis=1,keepdims=True)
# perform pca
variance_capture = 0.99
model = pca.pca(variance_capture)
reduced_X = model.compute_reduced(X)