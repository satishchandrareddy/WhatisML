# pca.py

import load_mnist
import numpy as np

class pca:
    def __init__(self,variance_capture):
        self.variance_capture = variance_capture

    def compute_reduced(self,X):
        # compute the singular value decomposition of X
        # return singular values 
        u, s, vh = np.linalg.svd(X)
        print("Original dimension: {}".format(np.size(s)))
        cumulative_variance = np.cumsum(np.square(s))
        total_variance = cumulative_variance[-1]
        print("Total variance: {}".format(total_variance))
        cumulative_variance_capture = cumulative_variance[cumulative_variance<=self.variance_capture*total_variance]
        reduced_dim = np.size(cumulative_variance_capture)
        print("Reduced dimension for {} variance capture: {}".format(self.variance_capture,reduced_dim))
        return np.expand_dims(s[0:reduced_dim],axis=1)*vh[0:reduced_dim,:]