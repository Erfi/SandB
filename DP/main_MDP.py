import numpy as np
import matplotlib.pyplot as plt

def getNeighbors(i,j,shape):
    '''
    returns an array of coordinates for the 
    neighbors of (i,j) if they exist in a grid with the 
    given shape. 
    A neighbor exists if we can reach it by
    one of the [up, left, down, right] actions.
    '''
    assert(i>=0 and i<shape[0] and j>=0 and j<shape[1])
    
    neighbors = []
    if (i-1 >= 0): #up
        neighbors.append((i-1,j))
    if (j-1 >= 0): #left
        neighbors.append((i, j-1))
    if (i+1 < shape[0]): #down
        neighbors.append((i+1, j))
    if (j+1 < shape[1]): #right
        neighbors.append((i, j+1))
    return neighbors

def printPolicy(states, goal):
    str_policy = "\n"
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            if ((i,j) in goal):
                str_policy += "GOL"
            else:
                neighbors = getNeighbors(i,j, states.shape)
                maxNeighborIndex = np.argmax([states[coord] for coord in neighbors])
                str_policy += printArrow((i,j), neighbors[maxNeighborIndex])
        str_policy += "\n"
    print(str_policy)
    
def printArrow(coord, maxCoord):
    diffVect = (maxCoord[0] - coord[0], maxCoord[1] - coord[1])
    if(diffVect == (1,0)):
        return " v "
    elif(diffVect == (-1,0)):
        return " ^ "
    elif(diffVect == (0,1)):
        return " > "
    elif(diffVect == (0,-1)):
        return " < "
    else:
        return "Oops in printArrow: diffVect: {}".format(diffVect)
    
    
def main():
    states = np.zeros((10,10))
    goal = [(0,0)]
    reward = -1.0
    discount = 0.9
    numIter = 10
    epsilon = 1e-2
    policyEval = False
    
    delta = epsilon + 1 #initializing to a number bigger than epsilon 
    while(delta > epsilon):
        co_states = np.zeros(states.shape)
        delta = 0 
        for i in range(states.shape[0]):
            for j in range(states.shape[1]):
                
                #define and skip goal state values
                if ((i,j) in goal):
                    continue
                v = states[i,j]
                neighbors = getNeighbors(i, j, states.shape)
                
                if(policyEval):#Policy Evaluation
                    sum_utils = 0 
                    for n in neighbors:
                        sum_utils += (1.0/len(neighbors) * 1.0 * (reward + discount*states[n]))
                    co_states[i,j] = sum_utils
                    delta = max(delta, np.abs(v - sum_utils))
                else:#Value Iteration
                    utilArray = [] 
                    for n in neighbors:
                        utilArray.append(1.0 * (reward + discount*states[n]))
                    co_states[i,j] = max(utilArray)
                    delta = max(delta, np.abs(v - co_states[i,j]))
                
        states = np.round(co_states.copy(), decimals=3)

    print(states)
    printPolicy(states, goal)
    plt.imshow(states)
    plt.colorbar()
    plt.show()
    
if __name__ == "__main__":
    main()