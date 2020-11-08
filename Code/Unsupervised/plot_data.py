# plot_data.py

import matplotlib.pyplot as plt

# this function is not part of the class is used for plottin

def plot_data2d(X,**kwargs):
    plt.figure()
    plt.plot(X[0,:],X[1,:],"bo",markersize=4)
    symbol = ["ks", "rs", "gs", "cs"]
    if "mean" in kwargs:
        mean = kwargs["mean"]
        ncluster = mean.shape[1]
        for count in range(ncluster):
            plt.plot(mean[0,count],mean[1,count],symbol[count],markersize=8)
    plt.xlabel("Relative Salary")
    plt.ylabel("Relative Purchases")
    plt.title("Data")
