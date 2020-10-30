# driver_kmeans.py

import create_data_cluster
import kmeans
import matplotlib.pyplot as plt
import numpy as np

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
# animation
model.plot_results_animation(X)
# plot results
kmeans.plot_data(X,mean=model.get_meansave()[0])
kmeans.plot_data(X,mean=model.get_meansave()[-1])
plt.show()