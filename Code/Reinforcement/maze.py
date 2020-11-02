# maze.py
#
# cliff problem
# -Sarsa using epsilon-greedy

from copy import deepcopy
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

class maze:
    def __init__(self,width,height,epsilon):
    	self.width = width
    	self.height = height
    	self.nstate = width*height
    	self.epsilon = epsilon
    	# 4 actions are fixed up, down, right, left
    	self.naction = 4

    def initialize_value(self):
    	# 4 actions fixed
    	self.value = np.zeros((self.nstate,self.naction))

    def initialize_state(self):
    	return self.grid2linear((0,0))
    	
    def get_action_epsgreedy(self,state):
    	# use epsilon-greedy approach
    	# pick random number in [0,1)
    	pick = np.random.rand()
    	# determine action
    	if pick<self.epsilon:
    		# random pick
    		return np.random.randint(self.naction)
    	else:
    		# pick randomly from max possible state-action value
    		return self.get_maxvalue_action(state)

    def get_maxvalue_action(self,state):
    	return np.random.choice(np.where(self.value[state,:]==self.value[state,:].max())[0])

    def grid2linear(self,state_grid):
    	return self.width*state_grid[0] + state_grid[1]

    def linear2grid(self,state_linear):
    	return (int(state_linear/self.width), state_linear%self.width)

    def transition(self,state,action):
        # returns reward, new state and new action
        # get new state
        state_grid = self.linear2grid(state)
        newstate_grid = (0,0)
        if action == 0: #up
            newstate_grid = (max(state_grid[0]-1,0),state_grid[1])
        if action == 1: #down
            newstate_grid = (min(state_grid[0]+1,self.height-1),state_grid[1])
        if action == 2: #left
            newstate_grid = (state_grid[0],max(state_grid[1]-1,0))
        if action == 3: #right
            newstate_grid = (state_grid[0],min(state_grid[1]+1,self.width-1))
        newstate = self.grid2linear(newstate_grid)
        newstate = self.check_wall(state,newstate)
        reward = -1
        newaction = self.get_action_epsgreedy(newstate)
        return reward,newstate,newaction

    def check_wall(self,state,newstate):
        # wall from top
        for panel in range(self.height-1):
            if (state == panel*self.width + self.width - 3) and (newstate == state+1):
                return state
            if (state == panel*self.width + self.width - 2) and (newstate == state-1):
                return state
        # wall from bottom   
        for panel in range(1,self.height):
            if (state == panel*self.width + self.width - 2) and (newstate == state+1):
                return state
            if (state == panel*self.width + self.width - 1) and (newstate == state-1):
                return state
        return newstate

    def get_action_prob(self,state):
    	prob = self.epsilon*np.ones((self.naction))/self.naction
    	idx = np.squeeze(np.where(self.value[state,:]==self.value[state,:].max()))
    	prob[idx] = prob[idx] + (1-self.epsilon)/np.size(idx)
    	return prob

    def isterminal(self,state):
    	return (state == self.nstate - 1)

    def sarsa(self,nepisode,alpha):
        ntransition = 1000
        reward_episode_avg = 0
        self.initialize_value()        
        self.valuesave = [deepcopy(self.value)]
        self.rewardsave = []
        for episode in range(nepisode):
            # initialize at start point
            state = self.initialize_state()
            action = self.get_action_epsgreedy(state)
            t = 0
            bterminal = False
            reward_sum = 0
            while (t<ntransition) and (not bterminal):
                reward,newstate,newaction = self.transition(state,action)
                #print("state: {} action: {} reward: {} newstate: {} newaction: {}".format(state,action,reward,newstate,newaction))
                self.value[state,action] += alpha*(reward + self.value[newstate,newaction] - self.value[state,action])
                bterminal = self.isterminal(newstate)
                state = deepcopy(newstate)
                action = deepcopy(newaction)
                t += 1
                reward_sum += reward
            #print("Episode: {} Reward: {}".format(episode+1,reward_sum))
            self.rewardsave.append(deepcopy(reward_sum))
            self.valuesave.append(deepcopy(self.value))

    def expectedsarsa(self,nepisode,alpha):
        ntransition = 1000
        reward_episode_avg = 0
        self.initialize_value()
        self.valuesave = [deepcopy(self.value)]
        self.rewardsave = []
        for episode in range(nepisode):
            # initialize at start point
            state = self.initialize_state()
            t = 0
            bterminal = False
            reward_sum = 0
            while (t<ntransition) and (not bterminal):
                action = self.get_action_epsgreedy(state)
                reward,newstate,_ = self.transition(state,action)
                #print("state: {} action: {} reward: {} newstate: {} newaction: {}".format(state,action,reward,newstate,newaction))
                prob = self.get_action_prob(state)
                self.value[state,action] += alpha*(reward + np.sum(self.value[newstate,:]*prob) - self.value[state,action])
                bterminal = self.isterminal(newstate)
                state = deepcopy(newstate)
                t += 1
                reward_sum += reward
            self.rewardsave.append(deepcopy(reward_sum))
            self.valuesave.append(deepcopy(self.value))

    def qlearning(self,nepisode,alpha):
        ntransition = 1000
        reward_episode_avg = 0
        self.initialize_value()
        self.valuesave = [deepcopy(self.value)]
        self.rewardsave = []
        for episode in range(nepisode):
            # initialize at start point
            state = self.initialize_state()
            t = 0
            bterminal = False
            reward_sum = 0
            while (t<ntransition) and (not bterminal):
                action = self.get_action_epsgreedy(state)
                reward,newstate,_ = self.transition(state,action)
                #print("state: {} action: {} reward: {} newstate: {} newaction: {}".format(state,action,reward,newstate,newaction))
                self.value[state,action] += alpha*(reward + self.value[newstate,self.get_maxvalue_action(newstate)] - self.value[state,action])
                bterminal = self.isterminal(newstate)
                state = deepcopy(newstate)
                t += 1
                reward_sum += reward
            self.rewardsave.append(deepcopy(reward_sum))
            self.valuesave.append(deepcopy(self.value))
    	    	
    def plot_reward(self):
        plt.figure()
        array_episode = np.arange(1,np.size(self.rewardsave)+1)
        array_reward = -np.array(self.rewardsave)
        plt.plot(array_episode,array_reward,"b-")
        plt.xlabel("Episode")
        plt.ylabel("Number of Moves")

    def plot_strategy_animation(self):
        nframe = len(self.valuesave)
        arrow_scale = 1.0
        # set up X and Y starting point for vectors
        X, Y = np.zeros((4*(self.nstate-1))),np.zeros((4*(self.nstate-1)))
        # update X and Y grids for origin of vectors
        for state in range(self.nstate-1):
            state2d = self.linear2grid(state)
            X[4*state] = state2d[1] + 0.5
            Y[4*state] = self.height - state2d[0] - 0.5
            X[4*state+1], X[4*state+2], X[4*state+3] = X[4*state], X[4*state], X[4*state]
            Y[4*state+1], Y[4*state+2], Y[4*state+3] = Y[4*state], Y[4*state], Y[4*state]
        # set up vector direction grids for each frame
        action_X = arrow_scale*np.array([0.0,0.0,-1.0,1.0])
        action_Y = arrow_scale*np.array([1.0,-1.0,0.0,0.0])
        Usave, Vsave = [], []
        for frame in range(nframe):
            U, V = np.zeros((4*(self.nstate-1))),np.zeros((4*(self.nstate-1)))
            for state in range(self.nstate-1):
                action = np.squeeze(np.where(self.valuesave[frame][state,:]==self.valuesave[frame][state,:].max()))
                #print("state: {}  action: {}".format(state,action))
                U[4*state + action] = action_X[action]
                V[4*state + action] = action_Y[action]
            Usave.append(deepcopy(U)), Vsave.append(np.array(V))

        # create initial image
        fig, ax = plt.subplots()
        # plot wall:
        ax.plot([self.width-2,self.width-2],[1,self.height],"k-")
        ax.plot([self.width-1,self.width-1],[0,self.height-1],"k-")
        ax.text(0.3,self.height-0.3,"Start",size=15)
        ax.text(self.width - 0.7, 0.5, "End", size=15)
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticklabels([])
        ax.set_xlim(0,self.width)
        ax.set_ylim(0,self.height)
        ax.grid(True)
        Q = ax.quiver(X, Y, Usave[0], Vsave[0], pivot='tail', color='r', units='inches')
        # frame update function
        def update_quiver(frame,Q,Usave,Vsave):
            Q.set_UVC(Usave[frame],Vsave[frame])
            return Q,

        # you need to set blit=False, or the first set of arrows never gets # cleared on subsequent frames
        ani = animation.FuncAnimation(fig, update_quiver, frames=nframe, fargs=(Q,Usave,Vsave),
                               repeat = False, interval=100, blit=False)
        fig.tight_layout()
        # uncomment to create mp4 
        # need to have ffmpeg installed on your machine - search for ffmpeg to get detaisl
        #ani.save('maze.mp4',writer='ffmpeg')
        plt.show()