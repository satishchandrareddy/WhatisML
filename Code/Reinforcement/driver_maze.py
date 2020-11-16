# driver_maze.py

import maze
import matplotlib.pyplot as plt
import numpy as np

# Things to try:
# Change random seed to get different random numbers: seed
# Change width and height of maze: width, height
# Change number of episodes: nepisode
seed = 11
width = 5
height = 5
nepisode = 50 
# (1) create model
np.random.seed(seed)
epsilon = 0.0
model = maze.maze(width,height,epsilon)
# (2) simulate
alpha = 1
model.qlearning(nepisode,alpha)
# (3) plot results
# number of steps for each episode
model.plot_steps()
# animation of strategy
model.plot_strategy_animation()
plt.show()