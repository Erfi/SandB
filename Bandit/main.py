import bandit as bndt
import numpy as np
import matplotlib.pyplot as plt

def experiment_epsilonGreedy(bandit, epsilon, numPulls):
    history = []
    arms = []
    for i in range(numPulls):
        arm = bandit.choose_eps_greedy(epsilon)
        reward = bandit.get_reward(arm)
        bandit.update_estimates(arm, reward)
        history.append(reward)
        arms.append(1 if arm==np.argmax(bandit.arm_values) else 0)
    return [np.array(history),np.array(arms)]

def experiment_softmax(bandit, temperature, numPulls):
    history = []
    arms = []
    for i in range(numPulls):
        arm = bandit.choose_softmax(temperature)
        reward = bandit.get_reward(arm)
        bandit.update_estimates(arm, reward)
        history.append(reward)
        arms.append(1 if arm==np.argmax(bandit.arm_values) else 0)
    return [np.array(history),np.array(arms)]

def main():
    print("---- Starting.... ----")
    Nexp = 100
    Npulls = 1000
    
    #=========== Epsilon Greedy Experiments ==========
    if(1):
        avg_outcome_eps0p0 = np.zeros(Npulls) 
        avg_outcome_eps0p01 = np.zeros(Npulls) 
        avg_outcome_eps0p1 = np.zeros(Npulls)
        avg_optimal_arm_eps0p0 = np.zeros(Npulls)
        avg_optimal_arm_eps0p01 = np.zeros(Npulls)
        avg_optimal_arm_eps0p1 = np.zeros(Npulls)

        for i in range(Nexp): 
            bandit = bndt.Bandit(10) #10 armed bandit 
            outcome_eps0p0, arms_eps0p0 = experiment_epsilonGreedy(bandit,0.0,Npulls)
            avg_outcome_eps0p0 += outcome_eps0p0
            avg_optimal_arm_eps0p0 += arms_eps0p0

            bandit = bndt.Bandit(10) #10 armed bandit 
            outcome_eps0p01, arms_eps0p01 = experiment_epsilonGreedy(bandit,0.01,Npulls)
            avg_outcome_eps0p01 += outcome_eps0p01
            avg_optimal_arm_eps0p01 += arms_eps0p01

            bandit = bndt.Bandit(10) #10 armed bandit 
            outcome_eps0p1, arms_eps0p1 = experiment_epsilonGreedy(bandit,0.1,Npulls)
            avg_outcome_eps0p1 += outcome_eps0p1
            avg_optimal_arm_eps0p1 += arms_eps0p1 

        avg_outcome_eps0p0 /= np.float(Nexp) 
        avg_outcome_eps0p01 /= np.float(Nexp) 
        avg_outcome_eps0p1 /= np.float(Nexp)
        avg_optimal_arm_eps0p0 /= np.float(Nexp)
        avg_optimal_arm_eps0p01 /= np.float(Nexp)
        avg_optimal_arm_eps0p1 /= np.float(Nexp)

        # plot results 
        plt.plot(avg_outcome_eps0p0,label="eps = 0.0") 
        plt.plot(avg_outcome_eps0p01,label="eps = 0.01") 
        plt.plot(avg_outcome_eps0p1,label="eps = 0.1") 
        plt.ylim(0,2) 
        plt.legend()
        plt.title('N-arm bandit problem simulation (N=10) using epsilon-greedy')
        plt.ylabel('Average Reward')
        plt.xlabel('Number of pulls/plays')

        plt.figure()

        plt.plot(avg_optimal_arm_eps0p0*100.0, label='eps = 0.0')
        plt.plot(avg_optimal_arm_eps0p01*100.0, label='eps = 0.01')
        plt.plot(avg_optimal_arm_eps0p1*100.0, label='eps = 0.1')
        plt.ylim(0,100)
        plt.legend(loc=0)
        plt.title('Average Percent Optimal Arm Chosen')
        plt.xlabel('Number of pulls/plays')
        plt.ylabel('Percent Optimal Arm')
        plt.show()
        
    #========== Softmax experiments ==========
    if(0):
        print('Softmax with different temperatures')
        #avg_outcome_eps = np.zeros(Npulls) 
        #avg_optimal_arm_eps = np.zeros(Npulls)
        avg_outcome_softmax0 = np.zeros(Npulls)
        avg_optimal_arm_softmax0 = np.zeros(Npulls)
        avg_outcome_softmax1 = np.zeros(Npulls)
        avg_optimal_arm_softmax1 = np.zeros(Npulls)
        avg_outcome_softmax2 = np.zeros(Npulls)
        avg_optimal_arm_softmax2 = np.zeros(Npulls)
        avg_outcome_softmax3 = np.zeros(Npulls)
        avg_optimal_arm_softmax3 = np.zeros(Npulls)
        
        for i in range(Nexp): 
            # bandit = bndt.Bandit(10) #10 armed bandit 
            # outcome_eps, arms_eps = experiment_epsilonGreedy(bandit,0.0,Npulls)
            # avg_outcome_eps += outcome_eps
            # avg_optimal_arm_eps += arms_eps
            
            bandit = bndt.Bandit(10) #10 armed bandit 
            outcome_softmax, arms_softmax = experiment_softmax(bandit,0.01,Npulls)
            avg_outcome_softmax0 += outcome_softmax
            avg_optimal_arm_softmax0 += arms_softmax
            
            bandit = bndt.Bandit(10) #10 armed bandit 
            outcome_softmax, arms_softmax = experiment_softmax(bandit,0.1,Npulls)
            avg_outcome_softmax1 += outcome_softmax
            avg_optimal_arm_softmax1 += arms_softmax
            
            bandit = bndt.Bandit(10) #10 armed bandit 
            outcome_softmax, arms_softmax = experiment_softmax(bandit,1,Npulls)
            avg_outcome_softmax2 += outcome_softmax
            avg_optimal_arm_softmax2 += arms_softmax
            
            bandit = bndt.Bandit(10) #10 armed bandit 
            outcome_softmax, arms_softmax = experiment_softmax(bandit,10,Npulls)
            avg_outcome_softmax3 += outcome_softmax
            avg_optimal_arm_softmax3 += arms_softmax
            
        
        # avg_outcome_eps /= np.float(Nexp)
        # avg_optimal_arm_eps /= np.float(Nexp)
        
        avg_outcome_softmax0 /= np.float(Nexp)
        avg_optimal_arm_softmax0 /= np.float(Nexp)
        
        avg_outcome_softmax1 /= np.float(Nexp)
        avg_optimal_arm_softmax1 /= np.float(Nexp)
        
        avg_outcome_softmax2 /= np.float(Nexp)
        avg_optimal_arm_softmax2 /= np.float(Nexp)
        
        avg_outcome_softmax3 /= np.float(Nexp)
        avg_optimal_arm_softmax3 /= np.float(Nexp)

        
        # plot results 
        # plt.plot(avg_outcome_eps,label="eps = 0.1") 
        
        plt.plot(avg_outcome_softmax0,label="temp = 0.01") 
        plt.plot(avg_outcome_softmax1,label="temp = 0.1")
        plt.plot(avg_outcome_softmax2,label="temp = 1")
        plt.plot(avg_outcome_softmax3,label="temp = 10")
        plt.ylim(0,2) 
        plt.legend()
        plt.title('N-arm bandit problem simulation (N=10) using softmax')
        plt.ylabel('Average Reward')
        plt.xlabel('Number of pulls/plays')

        plt.figure()

        # plt.plot(avg_optimal_arm_eps*100.0, label='eps = 0.1')
        plt.plot(avg_optimal_arm_softmax0*100.0, label='temp = 0.01')
        plt.plot(avg_optimal_arm_softmax1*100.0, label='temp = 0.1')
        plt.plot(avg_optimal_arm_softmax2*100.0, label='temp = 1')
        plt.plot(avg_optimal_arm_softmax3*100.0, label='temp = 10')
        plt.ylim(0,100)
        plt.legend(loc=0)
        plt.title('Average Percent Optimal Arm Chosen')
        plt.xlabel('Number of pulls/plays')
        plt.ylabel('Percent Optimal Arm')
        plt.show()
    
if __name__ == "__main__":
    main()