# onehot.py

import numpy as np

def onehot(Y,nclass):
    m = Y.shape[1]
    Y_onehot = np.zeros((nclass,m))
    for col in range(m):
        Y_onehot[int(Y[0,col]),col] = 1.0
    return Y_onehot

def onehot_inverse(Y_onehot):
    return np.expand_dims(np.argmax(Y_onehot,axis=0),axis=0)