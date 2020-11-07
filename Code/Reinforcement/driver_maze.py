# driver_maze.py

import maze
import matplotlib.pyplot as plt
import numpy as np

 # (1) create model
width = 5
height = 5
epsilon = 0.0
model = maze.maze(width,height,epsilon)
# (2) simulate
nepisode = 80
alpha = 1
model.qlearning(nepisode,alpha)
# (3) plot results
# number of steps for each episode
model.plot_steps()
# animation of strategy
model.plot_strategy_animation()
plt.show()