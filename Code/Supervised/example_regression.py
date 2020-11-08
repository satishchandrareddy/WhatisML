# example_classification.py

import matplotlib.pyplot as plt
import numpy as np

def example(nfeature,m):
	X = 0.5+3.5*np.random.rand(nfeature,m)
	Y = 0.3*X + 0.25 + 0.2*np.random.randn(nfeature,m)
	return X,Y

if __name__ == "__main__":
	nfeature = 1
	nsample = 200
	X,Y = example(nfeature,nsample)
	plt.figure()
	plt.scatter(np.squeeze(X),np.squeeze(Y),c="b",marker="o")
	plt.show()