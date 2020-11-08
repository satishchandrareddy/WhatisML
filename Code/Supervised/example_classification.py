# example_classification.py

import matplotlib.pyplot as plt
import numpy as np
import plot_results

def example(nfeature,m,case,nclass=2):
	X = 4*np.random.rand(nfeature,m)-2
	if case == "linear":
		Y = X[0,:] + X[1,:] - 0.25
	elif case == "quadratic":
		Y = X[1,:] - np.square(X[0,:]) + 1.5
	elif case == "cubic":
		Y = X[1,:] - np.power(X[0,:],3) - 2*np.power(X[0,:],2)+ 1.5
	elif case == "disk":
		Y = np.square(X[0,:])+np.square(X[1,:])-1
	elif case == "ring":
		Y = 1.25*np.sqrt(np.square(X[0,:])+np.square(X[1,:]))
		Y = np.fmod(Y,nclass)
	elif case == "band":
		Y = X[0,:] + X[1,:] 
		Y = np.fmod(Y,nclass)
	Y = np.maximum(Y,0.0)
	Y = np.round(Y)
	Y = np.minimum(Y,nclass-1)
	Y = np.expand_dims(Y,axis=0)
	return X,Y

if __name__ == "__main__":
	nfeature = 2
	m = 2000
	case = "ring"
	nclass = 2
	X,Y = example(nfeature,m,case,nclass)
	plot_results.plot_results_data(X,Y,nclass)
	plt.show()