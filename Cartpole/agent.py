'''
Erfan Azad
Date: 22 Feb 2017
'''

import gym
import collections
import copy
import numpy as np
import matplotlib.pyplot as plt
import time
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, normalization
from keras.optimizers import SGD,RMSprop,Nadam

class Agent(object):
    '''
    This class creates an agent to solve RL problems
    '''
    def __init__(self, env):
        '''
        Constuctor for the Agent class
        
        Args:
            env: The RL environments from openAI gym
        '''
        self.env = env
        self.memory = collections.deque(maxlen=10000)
        self.epsilon = 0.9
        self.epsilonDecayFactor = 0.99
        self.epsilonMin = 0.3
        self.gamma = 0.9                             # discount factor for Q-learning
        self.lr = 0.001                              # Learning Rate
        self.replayBatchSize = 200                   # Number of memories to use for training
        self.ET_decay = 0.99                         # Eligibility Trace decay factor
        self.ETrace = np.zeros((1,pow(2,env.reset().shape[0])))  # Initialize eligibility trace to all zeros
        self.model = self.build_model()
        self.printAgentProperties()
    
    def printAgentProperties(self):
        print("======= Agent =======\n")
        print("Env: {}\n".format(self.env))
        print("Initial Epsilon: {}\nMinimum Epsilon: {}\nEpsilon Decay Factor: {}\n".format(self.epsilon,
                                                                                            self.epsilonMin,
                                                                                            self.epsilonDecayFactor))
        print("Learning Rate(alpha): {}\nDiscount Factor(gamma): {}\nEligibility Trace Decay Factor: {}\nReplay Batch Size: {}\n".format(self.lr, self.gamma, self.ET_decay, self.replayBatchSize))
        print("=====================\n")
    
    def getBinNumber(self, state):
        '''
        Takes in an array of features, [f1,f2,f3,...fn],
        and returns a number between 0 to 2^(numFeatures)
        representing a bin number to be used in 
        eligibility trace calculation. 
        It breakes each feature into high (1) and low (0)
        and so creates a binary number [1 0 1 ... 1]
        then returns a desicmal version of that number.
        
        Args:
            state: An array of features [f1,f2,f3,...fn]
        
        Returns: 
            A number in range 0 to 2^(numFeatures)
        '''
        numFeatures = state.shape[0]
        # Limiting the number of bins to 2^20 = 1048576 bins b/c it sounds impractical to have infinite bins as numFeatures grows
        numFeatures = min(numFeatures, pow(2,20))
        high = self.env.observation_space.high
        low = self.env.observation_space.low
        mid_of_range = low + (high - low)/2
        bin_number = 0
        for i in range(numFeatures):
            feature_mask = 0 if state[i] < mid_of_range[i] else pow(2, (numFeatures-i-1))
            bin_number = bin_number|feature_mask

        return bin_number
        
    def incrementETrace(self, state, replace=True):
        '''
        Increments the eligibility trace for the 
        given state using either the 
        cumulative trace or replacing trace schemes
        
        Args:
            state: An array of features [f1,f2,f3,...fn]
            
            replace: Flag to which scheme to choose for ETrace
        '''
        index = self.getBinNumber(state)
        if (replace):
            self.ETrace[0][index] = 1
        else:
            self.ETrace[0][index] += 1
            
    def decayETrace(self):
        '''
        Decays all the elegibility traces by 
        a self.ET_decay factor
        '''
        self.ETrace *= (self.ET_decay*self.gamma)
        
        
    def build_model(self):
        '''
        Creates and Compiles a Keras model
        to be used as the function approximator 
        for the learning task.
        This model will take a state and predict the action-values
        for that state.
        '''
        model = keras.models.Sequential()
        model.add(Dense(64, input_dim=4, activation='tanh', init='he_uniform'))
        model.add(Dense(128, activation='tanh', init='he_uniform'))
        model.add(Dense(64, activation='tanh', init='he_uniform'))
        model.add(Dense(2, activation='linear', init='he_uniform'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.lr))
        return model
        
    def choose_action(self, state):
        '''
        Chooses actions from the state's action space
        in a epsilon-greedy fashion.
        
        Args:
            state: An array of features [f1,f2,f3,...fn]
        
        Returns:
            An action from state's action space.
        '''
        if (np.random.rand() > self.epsilon):
            action_values = self.model.predict(state)[0] #[Q(s,a1),Q(s,a2)]
            return np.argmax(action_values)
        else:
            return self.env.action_space.sample()
    
    def calculatePriority(self, memoryInstance):
        state, action, reward, next_state, done = memoryInstance
        Q_sa = self.model.predict(np.array(state).reshape(1,4))[0][action]    # Q(s,a)
        if not done:
            next_Qs = self.model.predict(np.array(next_state).reshape(1,4))[0]  # [Q_next(s_next,a1), Q_next(s_next,a2)]
            target = reward + self.gamma*max(next_Qs)         # This is what the Q(s,ai) should be (new estimate!)
        else:
            target = reward
        priority = abs(target - Q_sa)
        return priority
    
    
    def remember(self, memoryInstance):
        '''
        Records a touple of (state, action, reward, next_state, done)
        into the agents memory.
        '''
        priority = self.calculatePriority(memoryInstance)
        if (priority > 1.5): # If they are worthy!
            self.memory.append(memoryInstance)
    
    def doQLearningStep(self, memory, ETrace=True):
        '''
        Does one step of Q-Learning update
        '''
        state, action, reward, next_state, done = memory            # s, ai, r, s_next
        Q_estimate = self.model.predict(np.array(state).reshape(1,4))[0]    # [Q(s,a1), Q(s,a2)]
        if not done:
            next_Qs = self.model.predict(np.array(next_state).reshape(1,4))[0]  # [Q_next(s_next,a1), Q_next(s_next,a2)]
            target = reward + self.gamma*max(next_Qs)         # This is what the Q(s,ai) should be (new estimate!)
        else:
            target = reward
        Q_estimate[action] = target   # Updating the Q_estimate to use the better estimate for action ai
        if (ETrace):
            stateBinNumber = self.getBinNumber(state)             # bin number corresponding to state
            ET = self.ETrace[0][stateBinNumber]                   # Eligibility Trace for the bin
            K.set_value(self.model.optimizer.lr, self.lr*ET)      # update the learning rate to incorporate eligibility trace
        # Update the model parameters to fit the change    
            self.model.fit(np.array(state).reshape(1,4),
                           np.array(Q_estimate).reshape(1,2),
                           nb_epoch=1,
                           verbose=False)
        
        
    def replay(self, ETrace=True):
        '''
        This function is responsible for the learning 
        portion of the code.
        It is a modification of 2013 paper from DeepMind
        https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        
        It samples a batch from the agent's memory and updates
        the estimates according to the latest eligibility trace.
        
        It also decays the exploration probability after each episode
        when the replay function is called.
        
        Args:
            ETrace: Flag for whether or not use Eligibility Trace
        '''
        memorySampleIndices = np.random.choice(len(self.memory),
                                               size=min(len(self.memory),self.replayBatchSize),
                                               replace=False)
        for i in memorySampleIndices:
            self.doQLearningStep(self.memory[i], ETrace)
        self.epsilon = self.epsilonDecayFactor * self.epsilon if self.epsilon > self.epsilonMin else self.epsilon 
