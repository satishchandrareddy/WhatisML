# plot_data.py

import matplotlib.pyplot as plt
import numpy as np

# this function is not part of the class is used for plottin

def plot_data2d(X,**kwargs):
    plt.figure()
    plt.plot(X[0,:],X[1,:],color="b",marker="o",linestyle="None",markersize=4)
    color = get_colors()
    if "mean" in kwargs:
        mean = kwargs["mean"]
        ncluster = mean.shape[1]
        for count in range(ncluster):
            plt.plot(mean[0,count],mean[1,count],color=color[count], marker="s",linestyle="None",markersize=8)
    plt.xlabel("Relative Salary")
    plt.ylabel("Relative Purchases")
    plt.title("Data")

def get_colors():
    return ["k", "r", "g", "c", "m", "y", "tab:purple", "tab:brown", "tab:olive", "tab:pink", "tab:orange"]

def plot_data_mnist(X,Y=0):
    # create 5x5 subplot of mnist images
    nrow = 5
    ncol = 5
    npixel_width = 28
    npixel_height = 28
    fig,ax = plt.subplots(nrow,ncol,sharex="col",sharey="row")
    fig.suptitle("Images of Sample MNIST Digits")
    idx = 0
    for row in range(nrow):
        for col in range(ncol):
            digit_image = np.flipud(np.reshape(X[:,idx],(npixel_width,npixel_height)))
            ax[row,col].pcolormesh(digit_image,cmap="Greys")
            idx +=1