## Project description
Implementation of the TD3 - twin delayed DDPG algorithm for reinforcement learning ([original publication link](https://arxiv.org/pdf/1802.09477.pdf)), particularlly usefull for continuous action space-continuous state space problems.

The algorithm was tested on the [BipedalWalker-v3](https://gym.openai.com/envs/BipedalWalker-v2/) environment (even though the official documentation says that v2 is latest, it is deprecated!). We trained the agent on a high-performance GPU with CUDA, and after 550 episodes the following results were obtained:
![walk_demo](https://drive.google.com/uc?id=1y0_Z9uhuqt7hOb3m1wWrZzy1cKKR6NfV)

Project dependencies can be found in requirements.txt file, as usual.

## Todo
- To trully estimate how good is TD3 for this environment, we planned on repeating the training process 10-60 times, and from here we wanted to estimate the uncertainty of obtained reward for each agent-episode pair. We planned on constructing simple 95% confidence intervals for these quantities.
- Paper and presentation for this project are still being worked on.