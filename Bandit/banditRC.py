import numpy as np
import matplotlib.pyplot as plt

class Bandit(object):
    '''
    n-arm bandit class to be used as a test bed
    for **Stationary Problem** where the self.arm_values
    do not change throughout the learning process.
    This class uses Reinforcement-Comparison to select 
    an action.
    '''
    def __init__(self, numArms=10):
        '''
        Initializes the bandit with the given numner of 
        arms.
        '''
        #Actual values for each arm using normal distribution
        mu = 0
        sigma = 1
        self.numArms = numArms
        self.arm_values = np.random.normal(mu,sigma,numArms)
        #setting the reference reward
        self.refReward = 10.0
        #action preferences
        self.arm_prefs = np.zeros(numArms)
    
    def get_reward(self, action):
        '''
        Returns the reward for the given action.
        Using the actual value of the arm and a normal noise.s
        '''
        noise = np.random.normal(0,1)
        reward = self.arm_values[action] + noise
        return reward
        
    def choose_RC(self):
        '''
        Uses reinforcement comparison to 
        select among the availible actions.
        In this way the action with higher 
        'preferance' will have more probibility
        of getting selected. (similar to softmax)
        '''
        arm_probs = []
        den = np.sum(np.exp(self.arm_prefs))
        for i in range(self.numArms):
            arm_probs.append(np.exp(self.arm_prefs[i])/den)
        return np.random.choice(self.numArms, p=arm_probs)
        
    def update_estimates(self, arm, reward, alpha, beta):
        self.arm_prefs[arm] += beta * (reward - self.refReward)
        self.refReward += alpha * (reward - self.refReward)