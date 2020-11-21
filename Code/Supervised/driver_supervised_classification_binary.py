# driver_supervised_classification_binary.py

import example_classification
import matplotlib.pyplot as plt
import NeuralNetwork
import numpy as np
import Optimizer
import plot_results

# Things to try:
# Change random seed to get different random numbers: seed
# Change number of data samples: nsample
# Change data case: try case ="cubic", "disk"
# Change learning rate for optimization: learning_rate
# Change number of iterations: niteration
seed = 41
nsample = 1000
case = "quadratic"
learning_rate = 0.05
niteration = 100
# (1) Set up data
np.random.seed(seed)
nfeature = 2
nclass = 2
X,Y = example_classification.example(nfeature,nsample,case,nclass)
# (2) Define model
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(11,"tanh")
model.add_layer(8,"tanh")
model.add_layer(4,"tanh")
model.add_layer(1,"sigmoid")
# (3) Compile model and print summary
optimizer = Optimizer.Adam(learning_rate,0.9,0.999,1e-8)
model.compile("binarycrossentropy",optimizer)
# (4) Train model
history = model.fit(X,Y,niterations)
# (5) Results
# plot loss and accuracy
plot_results.plot_results_history(history,["loss"])
# plot data
plot_results.plot_results_data(X,Y,nclass)
# plot learning animation
plot_results.plot_results_classification_animation(X,Y,model,nclass)
plt.show()