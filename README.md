## Project description
Implementation of the TD3 - twin delayed DDPG algorithm for reinforcement learning ([original publication link](https://arxiv.org/pdf/1802.09477.pdf)), particularlly usefull for continuous action space-continuous state space problems.

The algorithm was tested on the [BipedalWalker-v3](https://gym.openai.com/envs/BipedalWalker-v2/) environment. In order to evaluate the variability of this algorithm, we trained 15 different agents on a high-performance GPU with CUDA for 550 episodes. We recorded the obtained reward by each agent, and obtained the following results:

![ci_plot](https://drive.google.com/uc?id=10C8y5Cd4TLgOPf2-ea22SUM9mXqdFB-q)

The learning process can be observed on the following video:
![run_simulation](https://drive.google.com/uc?id=1nPQ4f92XR9sbvwVg6HLMhGoBHQ5naSVn)

Technical details about the algorithm can be found in the acompanying report.