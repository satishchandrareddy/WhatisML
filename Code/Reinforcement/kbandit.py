# kbandit.py
# epsilon-greedy kbandit

from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

class kbandit:
    def __init__(self,k,epsilon):
        self.k = k
        self.q = np.zeros((k))
        self.qmean = np.sort(np.random.randn(k))
        self.na = np.zeros((k))
        self.epsilon = epsilon

    def update_q(self,a):
        reward = np.random.randn() + self.qmean[a]
        self.na[a] += 1
        self.q[a] = self.q[a] + (reward - self.q[a])/self.na[a]
        return reward

    def timestep(self,nstep):
        list_na = [deepcopy(self.na)]
        list_reward = [0]
        for step in range(nstep):
            epsilon_rand = np.random.rand()
            if epsilon_rand<self.epsilon:
                a = np.random.randint(np.size(self.q))
            else:
                a = np.random.choice(np.where(self.q==self.q.max())[0])
            reward = self.update_q(a)
            #print("self.na: {}".format(self.na))
            list_na.append(deepcopy(self.na/(step+1)))
            list_reward.append(reward)
        return np.array(list_na).T, np.array(list_reward)
    
def simulate(epsilon,k,nsim,nstep,seed):
    na_avg = np.zeros((k,nstep+1))
    reward_avg = np.zeros(nstep+1)
    np.random.seed(seed)
    for sim in range(nsim):
        kbandit_instance = kbandit(k,epsilon)
        na_sim, reward_sim = kbandit_instance.timestep(nstep)
        na_avg += na_sim/nsim
        reward_avg += reward_sim/nsim
    return na_avg, reward_avg

def run(list_eps,k,nsim,nstep,seed):
    list_na = []
    list_reward = []
    for epsilon in list_eps:
        print("Epsilon: {}".format(epsilon))
        na_sim, reward_sim = simulate(epsilon,k,nsim,nstep,seed)
        list_na.append(na_sim)
        list_reward.append(reward_sim)
    #print("list_na: {}".format(list_na))
    #print("list_reward: {}".format(list_reward))
    return list_na, list_reward

def plot_results_optimal(list_epsilon,list_results):
    plt.figure()
    label = ["b-","r-","k-"]
    plt.xlabel("Pull Number")
    plt.ylabel("Proportion of Pulls Optimal Bandit Chosen")
    for idx in range(len(list_epsilon)):
        array_epsilon = np.arange(np.size(list_results[idx][0,:]))
        str_label = "epsilon =" + str(list_epsilon[idx])
        plt.plot(array_epsilon,list_results[idx][-1,:],label[idx],label=str_label)
    plt.legend()

def plot_results_reward(list_epsilon,list_reward):
    plt.figure()
    label = ["b-", "r-", "k-"]
    plt.xlabel("Pull")
    plt.ylabel("Mean Reward")
    for idx in range(len(list_epsilon)):
        array_epsilon = np.arange(np.size(list_reward[idx]))
        str_label = "epsilon =" + str(list_epsilon[idx])
        plt.plot(array_epsilon,list_reward[idx],label[idx],label=str_label)
    plt.legend()

def plot_results_bar_animation1(epsilon,results):
    # extract information from list_results
    k = results.shape[0]
    nframe = results.shape[1]
    # create figure
    fig,ax = plt.subplots()
    plt.ylim(0,1)
    # create label for bar chart
    label = [str(i) for i in range(1,k+1)]
    bandit_number = np.arange(1,k+1)
    # generate frames
    list_frame = []
    for idx in range(nframe):
        title = ax.text(k/2,1.05,"Pull Number: {0}  Epsilon: {1}".format(idx,epsilon),size=10,ha="center", animated=True)
        xlabel("Bandit Number")
        ylabel("Proportion of Pulls")
        bars = ax.bar(bandit_number,results[:,idx],tick_label=label,color='blue', animated=True)
        print("pull: {}  results: {}".format(idx,results[:,idx]))
        frame = [title]
        frame.extend(list(bars))
        list_frame.append(frame)
    ani = animation.ArtistAnimation(fig,list_frame,interval=4,repeat=True,blit=False)
    plt.show()

def plot_results_bar_animation(list_epsilon,list_results):
    if len(list_results) != 2:
        print("Need two values of epsilon")
        return
    # extract information from list_results
    k = list_results[0].shape[0]
    nframe = list_results[0].shape[1]
    # create figure
    fig,(ax0,ax1) = plt.subplots(1,2)
    # create label for bar chart
    label = [str(i) for i in range(1,k+1)]
    bandit_number = np.arange(1,k+1)
    # generate frames
    list_frame = []
    for idx in range(nframe):
        title0 = ax0.text(k/2,1.05,"Pull Number: {0}  Epsilon: {1}".format(idx,list_epsilon[0]),size=10,ha="center", animated=True)
        bars0 = ax0.bar(bandit_number,list_results[0][:,idx],tick_label=label,color='blue', animated=True)
        ax0.set_ylim([0,1])
        ax0.set_xlabel("Bandit Number")
        ax0.set_ylabel("Proporation of Pulls")
        frame = [title0]
        frame.extend(bars0)
        title1 = ax1.text(k/2,1.05,"Pull Number: {0}  Epsilon: {1}".format(idx,list_epsilon[1]),size=10,ha="center", animated=True)
        bars1 = ax1.bar(bandit_number,list_results[1][:,idx],tick_label=label,color='blue', animated=True)
        ax1.set_ylim([0,1])
        ax1.set_xlabel("Bandit Number")
        frame.append(title1)
        frame.extend(bars1)
        list_frame.append(frame)
    ani = animation.ArtistAnimation(fig,list_frame,interval=70,repeat=True,blit=False)
    # uncomment to create mp4 
    # need to have ffmpeg installed on your machine - search for ffmpeg on internet for details
    # ani.save('bandit.mp4', writer='ffmpeg')
    return ani