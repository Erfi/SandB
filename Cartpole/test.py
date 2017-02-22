import sys
import getopt
import agent as AI
import numpy as np
import copy
import gym
import time

def argumentManager(argv):
    # Defalut values
    filename = None
    numEpisodes = 1
    render = False
    try:
        opts, args = getopt.getopt(argv[1:],"i:n:r:h",["input=","numEpisodes=", "render=", "help"])
    except getopt.GetoptError:
        print('test.py -i <input weight file (.h5)> -n <number of episodes> -r <render flag>')
        sys.exit()
        
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            filename = arg
        elif opt in ("-n", "--numEpisodes"):
            numEpisodes = int(arg)
        elif opt in ("-r","--render"):
            render = (arg == "True")
        elif opt in ("-h", "--help"):
            print("Usage: test.py -i <input weight file (.h5)> -n <number of episodes> -r <render flag>")
            sys.exit()
    
    if filename == None:
        print("Usage: test.py -i <input weight file (.h5)> -n <number of episodes> -r <render flag>")
        sys.exit(-1)
                
    return [filename, numEpisodes, render]


def main(argv):
    args = argumentManager(argv)
    inputWeightFile = args[0]
    numEpisodes = args[1]
    render = args[2]
    env = gym.make('CartPole-v0')
    agent = AI.Agent(env=env)
    agent.epsilon = 0.0 # No exploration for testing
    agent.model.load_weights(inputWeightFile)
    for episode in range(numEpisodes):
        state = env.reset()
        for t in range(10000):
            if (render):                                          
                env.render()
            action = agent.choose_action(np.array(state).reshape(1,4))  # Choose an action
            next_state, reward, done, info = env.step(action)           # Observe next_state, reward
            state = copy.deepcopy(next_state)                           # state = next_state
            if done:
                print("Episode {} finished after {} timesteps. memory size: {}. epsilon: {}".format(episode,t+1, len(agent.memory), agent.epsilon))
                break
            if (t >= 5000):
                print("Episode {} passed 5000 steps!".format(episode))
                break
    env.render(close=True)

if __name__ == "__main__":
    main(sys.argv)
