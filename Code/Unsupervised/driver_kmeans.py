# driver_kmeans.py

import create_data_cluster
import kmeans
import matplotlib.pyplot as plt
import numpy as np
import plot_data

# (1) generate data
# comment out seed line to generate different sets of random numbers
np.random.seed(21)
nfeature = 2
nsample = 200
ncluster = 3
std = 1
X,mean = create_data_cluster.create_data_cluster(nfeature,nsample,ncluster,std)
# (2) create model
model = kmeans.kmeans(nfeature,ncluster)
# (3) fit model
nepoch = 20
model.fit(X,nepoch)

# (4) plot results
model.plot_objective()
# plot initial data
plot_data.plot_data2d(X)
# plot initial data with initial means
plot_data.plot_data2d(X,mean=model.get_meansave()[0])
# plot final clusters
model.plot_cluster(X)
# animation
model.plot_results_animation(X)
plt.show()