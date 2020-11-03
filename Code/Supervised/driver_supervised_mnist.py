# driver_neuralnetwork_mnist.py

import load_mnist
import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
import Optimizer
import plot_results

# (1) Set up data
ntrain = 6000
nvalid = 1000
nclass = 10
Xtrain,Ytrain,Xvalid,Yvalid = load_mnist.load_mnist(ntrain,nvalid)
# (2) Define model
nfeature = Xtrain.shape[0]
np.random.seed(10)
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(128,"relu")
model.add_layer(nclass,"softmax")
# (3) Compile model
optimizer = Optimizer.Adam(0.02,0.9,0.999,1e-7)
model.compile("crossentropy",optimizer)
model.summary()
# (4) Train model
epochs = 40
history = model.fit(Xtrain,Ytrain,epochs)
# (5) Predictions and plotting
# plot data, loss, and animation of results
Ytrain_pred = model.predict(Xtrain)
plot_results.plot_data_mnist(Xtrain,Ytrain)
plot_results.plot_results_history(history,["loss"])
plot_results.plot_results_mnist_animation(Xtrain,Ytrain,Ytrain_pred,50)
plt.show()