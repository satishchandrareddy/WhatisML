# driver_neuralnetwork_binary.py

import example_classification
import matplotlib.pyplot as plt
import NeuralNetwork
import numpy as np
import Optimizer
import plot_results

# (1) Set up data
np.random.seed(41)
nfeature = 2
m = 1000
case = "quadratic"
nclass = 2
X,Y = example_classification.example(nfeature,m,case,nclass)
# (2) Define model
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(11,"tanh")
model.add_layer(8,"tanh")
model.add_layer(4,"tanh")
model.add_layer(1,"sigmoid")
# (3) Compile model and print summary
optimizer = Optimizer.Adam(0.05,0.9,0.999,1e-8)
model.compile("binarycrossentropy",optimizer)
# (4) Train model
epochs = 100
history = model.fit(X,Y,epochs)
# (5) Results
# plot loss and accuracy
plot_results.plot_results_history(history,["loss"])
# plot data
plot_results.plot_results_data(X,Y,nclass)
# plot heatmap in x0-x1 plane
# plot_results.plot_results_classification(X,Y,model,nclass)
plot_results.plot_results_classification_animation(X,Y,model,nclass)
plt.show()