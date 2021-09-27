# driver_reinforcement_kbandit.py

import kbandit
import matplotlib.pyplot as plt

# Things to try:
# Change random seed to get different random numbers: seed (integer)
# Change number of bandits: nbandit (between 2 and 12)
# Change epsilon - can compare 2 values: epsilon1 and epsilon2 (0<=epsilon<=1)
# Change number of pulls: npull
seed = 10
nbandit = 10
epsilon1 = 0
epsilon2 = 0.1
npull = 150
# simulate
list_epsilon = [epsilon1, epsilon2]
nsim = 1000
list_na, list_reward = kbandit.run(list_epsilon,nbandit,nsim,npull,seed)
# plot proportion optimal as a function of time
kbandit.plot_results_optimal(list_epsilon,list_na)
# plot average reward
kbandit.plot_results_reward(list_epsilon,list_reward)
# animation of bar charts
ani = kbandit.plot_results_bar_animation(list_epsilon[0:2],list_na[0:2])
plt.show()