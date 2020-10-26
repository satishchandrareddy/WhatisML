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
nepisode = 100
# sarsa
alpha = 1
#model.sarsa(nepisode,alpha)
# q-learning
model.qlearning(nepisode,alpha)
# expected sarsa
#model.expectedsarsa(nepisode,alpha)
# (3) plot results
model.plot_reward()
model.plot_strategy_animation()