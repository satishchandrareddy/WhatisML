# driver_maze.py

import kbandit
import matplotlib.pyplot as plt

# crease
list_epsilon = [0,0.1]
k = 10
nsim = 1000
nstep = 150
seed = 10
# simulate
list_na, list_reward = kbandit.run(list_epsilon,k,nsim,nstep,seed)
# plot proportion optimal as a function of time
kbandit.plot_results_optimal(list_epsilon,list_na)
# plot average reward
kbandit.plot_results_reward(list_epsilon,list_reward)
# animation of bar charts
kbandit.plot_results_bar_animation(list_epsilon[0:2],list_na[0:2])
plt.show()