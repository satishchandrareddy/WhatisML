# driver_neuralnetwork_multiclass.py

import example_classification
import matplotlib.pyplot as plt
import NeuralNetwork
import numpy as np
import Optimizer
import plot_results

# (1) Set up data
np.random.seed(11)
nfeature = 2
m = 2000
case = "quadratic"
nclass = 4
X,Y = example_classification.example(nfeature,m,case,nclass)
# (2) Define model
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(15,"tanh")
model.add_layer(11,"tanh")
model.add_layer(9,"tanh")
model.add_layer(6,"tanh")
model.add_layer(3,"tanh")
model.add_layer(nclass,"softmax")
# (3) Compile model
optimizer = Optimizer.Adam(0.02,0.9,0.999,1e-8)
model.compile("crossentropy",optimizer)
# (4) Train model
epochs = 100
history = model.fit(X,Y,epochs)
# (5) Results
# plot loss and accuracy
plot_results.plot_results_history(history,["loss"])
# plot data
plot_results.plot_results_data(X,Y,nclass)
# plot heatmap in x0-x1 plane
plot_results.plot_results_classification(X,Y,model,nclass)
plot_results.plot_results_classification_animation(X,Y,model,nclass)
plt.show()