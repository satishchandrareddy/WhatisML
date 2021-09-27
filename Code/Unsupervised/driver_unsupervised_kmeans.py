# driver_unsupervised_kmeans.py

import create_data_cluster
import kmeans
import matplotlib.pyplot as plt
import numpy as np
import plot_data

# Things to try:
# Change random seed to get different random numbers: seed
# Change number of data samples: nsample
# Change number of clusters: ncluster
# Change number of iterations if needed: niteration
seed = 21
nsample = 200
ncluster = 3
niteration = 20
# (1) generate data
# comment out seed line to generate different sets of random numbers
np.random.seed(seed)
nfeature = 2
std = 1
X,mean = create_data_cluster.create_data_cluster(nfeature,nsample,ncluster,std)
# (2) create model
model = kmeans.kmeans(nfeature,ncluster)
# (3) fit model
model.fit(X,niteration)
# (4) plot results
model.plot_objective()
# plot initial data
plot_data.plot_data2d(X,mean=model.get_meansave()[0])
# animation
ani = model.plot_results_animation(X)
plt.show()