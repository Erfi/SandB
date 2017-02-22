import numpy as np
import matplotlib.pyplot as plt

class Bandit(object):
    '''
    n-arm bandit class to be used as a test bed
    for **Stationary Problem** where the self.arm_values
    do not change throughout the learning process.
    This class can choose actions either by using:
    1) epsilon-greedy method
    2) softmax method
    '''
    def __init__(self, numArms=10):
        '''
        Initializes the bandit with the given numner of 
        arms.
        '''
        #Actual values for each arm using normal distribution
        mu = 0 #mean
        sigma = 1 #standard deviation
        self.numArms = numArms
        self.arm_values = np.random.normal(mu,sigma,numArms)
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
    
    def choose_softmax(self, temperature):
        '''
        Using Gibbs, or Boltzmann, distribution for
        softmax selection. 
        As temperature --> 0: choose_softmax = choose_eps_greedy,
        As tempurature --> inf: arm_probs[i] = 1/numArms
        '''
        arm_probs = []
        for i in range(len(self.arm_est_values)):
            arm_probs.append(np.exp(self.arm_est_values[i]/temperature)/np.sum(np.exp(self.arm_est_values/temperature)))
        return np.random.choice(len(self.arm_est_values), p=arm_probs)
        
    def update_estimates(self, arm, reward):
        '''
        Updates the estimated value for the given arm
        aswell as incrementing the number of times that
        arm has been used.
        '''
        self.K[arm] += 1
        alpha = 1./self.K[arm]
        self.arm_est_values[arm] += alpha * (reward - self.arm_est_values[arm])