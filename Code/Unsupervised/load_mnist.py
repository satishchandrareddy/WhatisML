# load_mnist.py

import numpy as np
import pandas as pd
from pathlib import Path

def load_mnist(ntrain,nvalid):
	# read data from train files
	root_dir = Path(__file__).resolve().parent.parent
	dftrain1 = pd.read_csv(root_dir / "Data_MNIST/MNIST_train_set1_30K.csv")
	dftrain2 = pd.read_csv(root_dir / "Data_MNIST/MNIST_train_set2_30K.csv")
	# get labels - concatenate and convert to row vector
	Ytrain1 = dftrain1["label"].values
	Ytrain2 = dftrain2["label"].values
	Ytrain = np.concatenate((Ytrain1,Ytrain2),axis=0)
	Ytrain = np.reshape(Ytrain,(1,Ytrain.shape[0]))
	# remove label column
	dftrain1 = dftrain1.drop(columns="label")
	dftrain2 = dftrain2.drop(columns="label")
	# create feature matrix from remaining data - divide by 255
	Xtrain1 = dftrain1.values/255
	Xtrain2 = dftrain2.values/255
	Xtrain = np.concatenate((Xtrain1,Xtrain2),axis=0)
	Xtrain = Xtrain.T
	# get requested number of training data samples
	Xtrain = Xtrain[:,:ntrain]
	Ytrain = Ytrain[:,:ntrain]
	# get validation data
	dfvalid = pd.read_csv(root_dir / "Data_MNIST/MNIST_valid_10K.csv")
	Yvalid = dfvalid["label"].values
	Yvalid = np.reshape(Yvalid,(1,Yvalid.shape[0]))
	dfvalid = dfvalid.drop(columns="label")
	Xvalid = dfvalid.values.T/255
	# get requested number of validation data samples
	Xvalid = Xvalid[:,:nvalid]
	Yvalid = Yvalid[:,:nvalid]
	print("Xtrain.shape: {} - Ytrain.shape: {}".format(Xtrain.shape,Ytrain.shape))
	print("Xvalid.shape: {} - Yvalid.shape: {}".format(Xvalid.shape,Yvalid.shape))
	return Xtrain,Ytrain,Xvalid,Yvalid

if __name__ == "__main__":
	Xtrain,Ytrain,Xvalid,Yvalid = load_mnist(6000,1000)