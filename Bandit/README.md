# barto
Code for simulations from "Reinforcement Learning by Richard S. Sutton and Andrew G. Barto"

The code is a modified version of the code here: http://blog.thedataincubator.com/2016/07/multi-armed-bandits-2/

***

*bandit.py:* Bandit class for *Stationary* version of the problem where the actual value of the actions don't change throughout the simulation.

*main.py:* Defines the "experiment" functions and runs the simulation for epsilon-greedy and softmax actions selection methods using bandit.py

***

*bandit2.py:* Bandit class for *non-stationary* version of the problem where the actual value of the actions take random walk in each step throughout the simulaiton.

*main2.py:* Defines the "experiment" funcitons and runs the simulaiton for the non-stationary problem of bandit2 class using epsilon-greedy action selsection.

***

*banditRC.py:* Bandit class for *stationary* version of the n-arm bandit problem. This class uses *Reinforcement Comparison* for its action selection.

*mainRC.py:* Defines the "experiment" functions and runs to simulation for the banditRC class.

