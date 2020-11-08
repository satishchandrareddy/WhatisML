# driver_supervised_linearregression.py

import example_regression
import matplotlib.pyplot as plt
import NeuralNetwork
import numpy as np
import Optimizer
import plot_results

# (1) Set up data
np.random.seed(100)
nfeature = 1
nsample = 500
X,Y = example_regression.example(nfeature,nsample)
# (2) define model
model = NeuralNetwork.NeuralNetwork(nfeature)
model.add_layer(1,"linear")
# (3) set initial guess for W and b
initial_param = [{"W": -0.2, "b":1.4}]
model.set_param(initial_param)
# (4) define loss function and optimizer
optimizer = Optimizer.GradientDescent(0.1)
model.compile("meansquarederror",optimizer)
# (4) Train model
niteration = 100
history = model.fit(X,Y,niteration)
# (5) Results
# plot loss 
plot_results.plot_results_history(history,["loss"])
# plot data and final machine learning solution
plot_results.plot_results_linear(X,Y,model)
# animation
plot_results.plot_results_linear_animation(X,Y,model)
plt.show()