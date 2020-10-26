# create_data.py

import matplotlib.pyplot as plt
import numpy as np
import kmeans

def create_data_cluster(nfeature,nsample,ncluster,std=1):
	# generate means
	mean = 3*np.random.randn(nfeature,ncluster)
	npercluster = int(np.ceil(nsample/ncluster))
	X = np.zeros((nfeature,0))
	for i in range(ncluster):
		X = np.concatenate((X,mean[:,i:i+1]+std*np.random.randn(nfeature,npercluster)),axis=1)
	return X,mean
	
if __name__ == "__main__":
	nfeature = 2
	nsample = 30
	ncluster = 3
	std = 0.5
	X,mean = create_data_cluster(nfeature,nsample,ncluster,std)
	kmeans.plot_data(X,mean=mean)
	plt.show()