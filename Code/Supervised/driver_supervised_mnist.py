# driver_supervised_mnist.py

import load_mnist
import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
import Optimizer
import plot_results

# Things to try:
# Change random seed to get different random numbers: seed
# Change number of training data samples: ntrain up to 60000
# Change number of validation data samples: nvalid up to 10000
# Change learning rate for optimization: learning_rate
# Change number of iterations: niterations
seed = 10
ntrain = 6000
nvalid = 1000
learning_rate = 0.02
niteration = 40
# (1) Set up data
nclass = 10
Xtrain,Ytrain,Xvalid,Yvalid = load_mnist.load_mnist(ntrain,nvalid)
# (2) Define model
nfeature = Xtrain.shape[0]
np.random.seed(seed)
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(128,"relu")
model.add_layer(nclass,"softmax")
# (3) Compile model
optimizer = Optimizer.Adam(learning_rate,0.9,0.999,1e-7)
model.compile("crossentropy",optimizer)
model.summary()
# (4) Train model
history = model.fit(Xtrain,Ytrain,niteration)
# (5) Predictions and plotting
# plot data, loss, and animation of results
Yvalid_pred = model.predict(Xvalid)
plot_results.plot_data_mnist(Xtrain,Ytrain)
plot_results.plot_results_history(history,["loss"])
plot_results.plot_results_mnist_animation(Xvalid,Yvalid,Yvalid_pred,25)