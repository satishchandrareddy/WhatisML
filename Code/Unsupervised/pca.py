# pca.py

import load_mnist
import numpy as np

class pca:
    def __init__(self,variance_capture):
        self.variance_capture = variance_capture

    def compute_reduced(self,X):
        # return X projected onto the reduced dimensional space
        # number of dimensions
        ndim = X.shape[0]
        print("Original dimension: {}".format(ndim))
        # comopute SVD
        u, s, vh = np.linalg.svd(X)
        # cumpute cumulative sum of squares of variance
        cumulative_variance = np.cumsum(np.square(s))
        total_variance = cumulative_variance[-1]
        print("Total variance: {}".format(total_variance))
        # determine number of dimension to capture variance_capture proportion of variance
        cumulative_variance_prop = cumulative_variance/total_variance
        cumulative_variance_capture = cumulative_variance_prop[cumulative_variance_prop<=self.variance_capture]
        # add 1 to reduced dimension to make sure we have at least variance_capture proportion of variance
        reduced_dim = np.size(cumulative_variance_capture)
        if reduced_dim == 0:
            reduced_dim = 1
        elif reduced_dim >0:
            if cumulative_variance_prop[reduced_dim-1]<self.variance_capture:
                reduced_dim += 1
        print("Reduced dimension: {} - variance capture proportion: {}".format(reduced_dim,cumulative_variance_prop[reduced_dim-1]))
        return np.dot(u[:,0:reduced_dim],np.expand_dims(s[0:reduced_dim],axis=1)*vh[0:reduced_dim,:])