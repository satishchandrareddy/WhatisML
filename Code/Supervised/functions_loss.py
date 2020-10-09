# functions_loss.py

import numpy as np
import onehot

def loss(loss_fun,A,Y):
	m = A.shape[1]
	if loss_fun == "meansquarederror":
		return np.sum(np.square(A-Y))/m
	elif loss_fun == "binarycrossentropy":
		return -np.sum(Y*np.log(A+1e-16)+(1-Y)*np.log(1-A+1e-16))/m
	elif loss_fun == "crossentropy":
		nclass = A.shape[0]
		return -np.sum(onehot.onehot(Y,nclass)*np.log(A+1e-16))/m

def loss_der(loss_fun,A,Y):
	m = A.shape[1]
	if loss_fun == "meansquarederror":
		return 2*(A-Y)/m
	elif loss_fun == "binarycrossentropy":
		return (-Y/(A+1e-16) + (1-Y)/(1-A+1e-16))/m
	elif loss_fun == "crossentropy":
		nclass = A.shape[0]
		return -onehot.onehot(Y,nclass)/(A+1e-16)/m