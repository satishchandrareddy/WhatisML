# example_regression.py

import matplotlib.pyplot as plt
import numpy as np
import plot_results

def example(nfeature,m):
	X = 0.5+3.5*np.random.rand(nfeature,m)
	Y = np.absolute(0.3*X + 0.25 + 0.2*np.random.randn(nfeature,m))
	return X,Y

if __name__ == "__main__":
	np.random.seed(100)
	nfeature = 1
	nsample = 500
	X,Y = example(nfeature,nsample)
	plot_results.plot_results_linear(X,Y)
	plt.show()