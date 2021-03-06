import sys
import getopt
import agent as AI
import numpy as np
import copy
import gym
import time

def argumentManager(argv):
    # Defalut values
    numEpisodes = 1
    render = False
    try:
        opts, args = getopt.getopt(argv[1:],"n:r:h",["numEpisodes=", "render", "help"])
    except getopt.GetoptError:
        print('train.py -n <number of episodes>')
        
    for opt, arg in opts:
                if opt in ("-n", "--numEpisodes"):
                    numEpisodes = int(arg)
                elif opt in ("-r", "--render"):
                    render = (arg=="True")
                elif opt in ("-h", "--help"):
                    print('Usage: train.py -n <number of episodes>')
                    sys.exit()
    return [numEpisodes, render]

def main(argv):
    args = argumentManager(argv)
    numEpisodes = args[0]
    render = args[1]
    env = gym.make('CartPole-v0')
    agent = AI.Agent(env=env)
    for episode in range(numEpisodes):
        state = env.reset()
        for t in range(5000):
            if(render):
                env.render()
            agent.incrementETrace(state, replace=True)
            action = agent.choose_action(np.array(state).reshape(1,4))  # Choose an action
            next_state, reward, done, info = env.step(action)           # Observe next_state, reward
            reward = -2000/t if done else reward*0.1*t                  # heavily penalize the failing reward
            memoryInstance = (state, action, reward, next_state, done)  # create a memory instance
            agent.remember(memoryInstance)                              # remember SARS + done
            state = copy.deepcopy(next_state)                           # state = next_state
            if done:
                print("Episode {} finished after {} timesteps. memory size: {}. epsilon: {}".format(episode,t+1, len(agent.memory), agent.epsilon))
                break
        if(episode%50 == 0):
            print("Saving the model...\n")
            agent.model.save_weights("./weights/weight_minEps{}_numEpisodes{}.h5".format(agent.epsilonMin, numEpisodes))
        agent.decayETrace()
        agent.replay(ETrace=True)
    if(render):    
        env.render(close=True)
if __name__ == "__main__":
    main(sys.argv)
