# Optimizer.py

import numpy as np

class Optimizer_Base:
    def __init__(self):
        pass

    def update(self):
        pass

class GradientDescent(Optimizer_Base):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

    def update(self,gradient):
        return -self.learning_rate*gradient

class Momentum(Optimizer_Base):
    def __init__(self,learning_rate,beta):
        self.learning_rate = learning_rate
        self.beta = beta
        self.v = 0

    def update(self,gradient):
        self.v = self.beta*self.v + gradient
        return -self.learning_rate*self.v

class RmsProp(Optimizer_Base):
    def __init__(self,learning_rate,beta,epsilon):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = 0

    def update(self,gradient):
        self.v = self.beta*self.v + (1-self.beta)*np.square(gradient)
        return -self.learning_rate*gradient/(np.sqrt(self.v)+self.epsilon)

class Adam(Optimizer_Base):
    def __init__(self,learning_rate,beta1,beta2,epsilon):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0

    def update(self,gradient):
        self.m = self.beta1*self.m + (1-self.beta1)*gradient
        self.v = self.beta2*self.v + (1-self.beta2)*np.square(gradient)
        return -self.learning_rate*self.m/(np.sqrt(self.v)+self.epsilon)