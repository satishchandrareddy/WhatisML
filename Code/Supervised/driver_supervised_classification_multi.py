# driver_supervised_classification_multi.py

import example_classification
import matplotlib.pyplot as plt
import NeuralNetwork
import numpy as np
import Optimizer
import plot_results

# Things to try:
# Change random seed to get different random numbers: seed (integer)
# Change number of data samples: nsample
# Change data case: try case = "quadratic", "cubic"
# Change number of classes: nclass (between 2 and 4)
# Change learning rate for optimization: learning_rate >0
# Change number of iterations: niterations
seed = 11
nsample = 2000
case = "quadratic"
nclass = 4
learning_rate = 0.02
niteration = 100
# (1) Set up data
np.random.seed(seed)
nfeature = 2
X,Y = example_classification.example(nfeature,nsample,case,nclass)
# (2) Define model
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(15,"tanh")
model.add_layer(11,"tanh")
model.add_layer(9,"tanh")
model.add_layer(6,"tanh")
model.add_layer(3,"tanh")
model.add_layer(nclass,"softmax")
# (3) Compile model
optimizer = Optimizer.Adam(learning_rate,0.9,0.999,1e-8)
model.compile("crossentropy",optimizer)
# (4) Train model
history = model.fit(X,Y,niteration)
# (5) Results
# plot loss and accuracy
plot_results.plot_results_history(history,["loss"])
# plot data
plot_results.plot_results_data(X,Y,nclass)
# plot final heatmap
plot_results.plot_results_classification(X,Y,model,nclass)
# plot animation
plot_results.plot_results_classification_animation(X,Y,model,nclass)
plt.show()