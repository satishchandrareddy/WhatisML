# functions_activation.py

import numpy as np

def activation(activation_fun,Z):
	LIM = 50
	if activation_fun == "linear":
		return Z
	elif activation_fun == "sigmoid":
		ZLIM = np.maximum(Z,-LIM)
		return 1/(1+np.exp(-ZLIM))
	elif activation_fun == "relu":
		return np.maximum(Z,0.0)
	elif activation_fun == "tanh":
		return np.tanh(Z)
	elif activation_fun == "softplus":
		ZLIM = np.maximum(Z,-LIM)
		return ZLIM + np.log(np.exp(-ZLIM)+1)
	elif activation_fun == "softmax":
		Zmax_col = np.amax(Z,axis=0)
		numerator = np.exp(Z-Zmax_col)
		denominator = np.sum(numerator,axis=0)
		return numerator/denominator

def activation_der(activation_fun,A,grad_A_L):
	if activation_fun == "linear":
		return grad_A_L*np.ones(A.shape)
	elif activation_fun == "sigmoid":
		return grad_A_L*(A - np.square(A))
	elif activation_fun == "relu":
		return grad_A_L*(np.piecewise(A,[A<0,A>=0],[0,1]))
	elif activation_fun == "tanh":
		return grad_A_L*(1 - np.square(A))
	elif activation_fun == "softplus":
		return grad_A_L*(1 - np.exp(-A))
	elif activation_fun == "softmax":
		return grad_A_L*A - A*np.sum(grad_A_L*A,axis=0)