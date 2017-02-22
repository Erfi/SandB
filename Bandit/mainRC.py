import banditRC as bndt
import bandit as bndt_eps
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

def experiment_RC(bandit, numPulls, alpha, beta):
    history = []
    arms = []
    for i in range(numPulls):
        arm = bandit.choose_RC()
        reward = bandit.get_reward(arm)
        history.append(reward)
        arms.append(1 if arm==np.argmax(bandit.arm_values) else 0)
        bandit.update_estimates(arm, reward, alpha, beta)
    return [np.array(history), np.array(arms)]

def main():
    print("---- Starting.... ----")
    Nexp = 1000
    Npulls = 2000
    
    #=========== Epsilon Greedy Experiments (Nonstationary) ==========
    if(1):
        avg_outcome_RC1 = np.zeros(Npulls) 
        avg_optimal_arm_RC1 = np.zeros(Npulls)
        
        avg_outcome_eps1 = np.zeros(Npulls) 
        avg_optimal_arm_eps1 = np.zeros(Npulls)
        


        for i in range(Nexp): 
            bandit = bndt.Bandit(10) #10 armed bandit
            outcome_RC, arms_RC = experiment_RC(bandit,Npulls, alpha=0.1, beta=0.2)
            avg_outcome_RC1 += outcome_RC
            avg_optimal_arm_RC1 += arms_RC
            
            bandit = bndt_eps.Bandit(10) #10 armed bandit
            outcome_eps1, arms_eps1 = experiment_epsilonGreedy(bandit, 0.1, Npulls)
            avg_outcome_eps1 += outcome_eps1
            avg_optimal_arm_eps1 += arms_eps1
            

        avg_outcome_RC1 /= np.float(Nexp) 
        avg_optimal_arm_RC1 /= np.float(Nexp) 
        
        avg_outcome_eps1 /= np.float(Nexp) 
        avg_optimal_arm_eps1 /= np.float(Nexp)


        # plot results 
        plt.plot(avg_outcome_RC1,label="RC: a=0.1 b=0.2")
        plt.plot(avg_outcome_eps1,label="Eps: eps=0.1 a=1/k")
        plt.legend(loc=0)
        plt.title('Average Reward: Eps-Greedy vs Reinf. Comp. (Stationary Problem)')
        plt.ylabel('Average Reward')
        plt.xlabel('Number of pulls/plays')

        plt.figure()

        plt.plot(avg_optimal_arm_RC1*100.0, label='RC a=0.1 b=0.2')
        plt.plot(avg_optimal_arm_eps1*100.0, label='Eps eps=0.1 a=1/k')
        plt.ylim(0,100)
        plt.legend(loc=0)
        plt.title('Average %Optimal Arm Chosen: Eps-Greedy vs Reinf. Comp.(Stationary Problem)')
        plt.xlabel('Number of pulls/plays')
        plt.ylabel('Percent Optimal Arm')
        plt.show()
    
if __name__ == "__main__":
    main()