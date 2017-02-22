import bandit2 as bndt
import numpy as np
import matplotlib.pyplot as plt

def experiment_epsilonGreedy(bandit, epsilon, numPulls, alpha=None):
    history = []
    arms = []
    for i in range(numPulls):
        arm = bandit.choose_eps_greedy(epsilon)
        reward = bandit.get_reward(arm)
        history.append(reward)
        arms.append(1 if arm==np.argmax(bandit.arm_values) else 0)
        bandit.update_estimates(arm, reward, alpha)
    return [np.array(history),np.array(arms)]

def main():
    print("---- Starting.... ----")
    Nexp = 1000
    Npulls = 5000
    
    #=========== Epsilon Greedy Experiments (Nonstationary) ==========
    if(1):
        avg_outcome_eps_contAlpha = np.zeros(Npulls) 
        avg_outcome_eps_varAlpha = np.zeros(Npulls) 
        avg_optimal_arm_eps_constAlpha = np.zeros(Npulls)
        avg_optimal_arm_eps_varAlpha = np.zeros(Npulls)

        for i in range(Nexp): 
            bandit = bndt.Bandit(10) #10 armed bandit constant alpha=0.1
            outcome_eps, arms_eps = experiment_epsilonGreedy(bandit,0.1,Npulls, alpha=0.1)
            avg_outcome_eps_contAlpha += outcome_eps
            avg_optimal_arm_eps_constAlpha += arms_eps

            bandit = bndt.Bandit(10) #10 armed bandit variable alpha=1/k
            outcome_eps, arms_eps = experiment_epsilonGreedy(bandit,0.1,Npulls)
            avg_outcome_eps_varAlpha += outcome_eps
            avg_optimal_arm_eps_varAlpha += arms_eps 

        avg_outcome_eps_contAlpha /= np.float(Nexp) 
        avg_outcome_eps_varAlpha /= np.float(Nexp) 
        avg_optimal_arm_eps_constAlpha /= np.float(Nexp)
        avg_optimal_arm_eps_varAlpha /= np.float(Nexp)

        # plot results 
        plt.plot(avg_outcome_eps_contAlpha,label="Constant Alpha") 
        plt.plot(avg_outcome_eps_varAlpha,label="Variable Alpha") 
        # plt.ylim(0,2) 
        plt.legend(loc=0)
        plt.title('10-arm bandit simulaiton using epsilon = 0.1 (non-stationary problem)')
        plt.ylabel('Average Reward')
        plt.xlabel('Number of pulls/plays')

        plt.figure()

        plt.plot(avg_optimal_arm_eps_constAlpha*100.0, label='Constant Alpha')
        plt.plot(avg_optimal_arm_eps_varAlpha*100.0, label='Variable Alpha')
        plt.ylim(0,100)
        plt.legend(loc=0)
        plt.title('Average Percent Optimal Arm Chosen (non-stationary problem)')
        plt.xlabel('Number of pulls/plays')
        plt.ylabel('Percent Optimal Arm')
        plt.show()
    
if __name__ == "__main__":
    main()