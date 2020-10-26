# driver_supervised_linearregression.py

import matplotlib.pyplot as plt
import NeuralNetwork
import numpy as np
import Optimizer
import plot_results

# (1) Set up data
np.random.seed(100)
m = 1000
X = np.random.rand(1,m)
Y = 0.5*X + 0.25
Y = Y + 0.1*np.random.randn(m)
# (2) Define model
model = NeuralNetwork.NeuralNetwork(1)
model.add_layer(1,"linear")
# (3) Compile model
optimizer = Optimizer.GradientDescent(0.5)
model.compile("meansquarederror",optimizer)
# (4) Train model
epochs = 50
history = model.fit(X,Y,epochs)
# (5) Results
# plot loss 
plot_results.plot_results_history(history,["loss"])
# plot results 
plot_results.plot_results_linear(X,Y,model)
# animation
plot_results.plot_results_linear_animation(X,Y,model)
plt.show()