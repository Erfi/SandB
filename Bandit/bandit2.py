import numpy as np
import matplotlib.pyplot as plt

class Bandit(object):
    '''
    n-arm bandit class to be used as a test bed
    for **Non-Stationary Problem** where the self.arm_values
    do change throughout the learning process.
    '''
    def __init__(self, numArms=10):
        '''
        Initializes the bandit with the given numner of 
        arms.
        '''
        #Actual values for each arm using same random number for all arms
        mu = 0
        sigma = 1
        self.numArms = numArms
        self.arm_values = np.ones(numArms)*np.random.normal(mu,sigma)
        #Number of times each arm is played
        self.K = np.zeros(numArms)
        #Estimated value for each arm
        self.arm_est_values = np.zeros(numArms)
    
    def get_reward(self, action):
        '''
        Returns the reward for the given action.
        Using the actual value of the arm and a normal noise.s
        '''
        noise = np.random.normal(0,1)
        reward = self.arm_values[action] + noise
        return reward
    
    def choose_eps_greedy(self, epsilon):
        '''
        Uses epsilon to determine the action.
        This is where we choose between exploration
        and exploitation.
        '''
        rand_num = np.random.random()
        if  epsilon > rand_num:
            return np.random.randint(self.numArms)
        else:
            return np.argmax(self.arm_est_values)
        
    def update_estimates(self, arm, reward, alpha=None):
        self.K[arm] += 1
        if(alpha):
            alpha = alpha
        else:
            alpha = 1./self.K[arm]
        self.arm_est_values[arm] += alpha * (reward - self.arm_est_values[arm])
        
        #Change the self.arm_values based on an individual
        #random walk in order to create a non-stationary problem
        noise = np.random.normal(0,1,self.numArms)
        self.arm_values += noise
        
        