import csv
import gym
import torch
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt

from Agent import Agent
from os import path

# HYPERPARAMETERS
gamma = 0.99         # discount factor for rewards
learningRate = 3e-4  # learning rate for actor and critic networks
tau = 0.005          # tracking parameter used to update target networks slowly
actionSigma = 0.1    # contributes noise to deterministic policy output
trainingSigma = 0.2  # contributes noise to target actions
trainingClip = 0.5   # clips target actions to keep them close to true actions
miniBatchSize = 100  # how large a mini-batch should be when updating
policyDelay = 2      # how many steps to wait before updating the policy
resume = True        # resume from previous checkpoint if possible?
render = False       # render out the environment?
episode_limit = 550  # limiting the number of episodes, including pretrained episodes

envName = "BipedalWalker-v3"

def train(trials = 1, suffix = "", periodicSaving = False, period = 100):
    """
    Trains the agent.
    Parameters:
        trials - number of different trials to be executed. We perform more than 1 trial to 
                 track the reward for every episode and trial pair in order to estimate how 
                 good is the TD3 algorithm for this environment.
        
        suffix - suffix for the environment file, used to give IDs to saved agents.

        periodicSaving - whether or not to save the learned model one additional time. Used
                         to reproduce the learning history for presentation (still to be done).
    
        period - period for additional agent saving. 
    """
    
    for trial in range(0, trials):
        env = gym.make(envName)
        env.name = envName + suffix + "_" + str(trial)
        csvName = env.name + '-data.csv'
        agent = Agent(env, learningRate, gamma, tau, resume)
        step = 0
        runningReward = None

        # determine the last episode if we have saved training in progress
        numEpisode = 0

        if path.exists(csvName):
            fileData = list(csv.reader(open(csvName)))
            lastLine = fileData[-2]
            numEpisode = int(lastLine[0])
            runningReward = float(lastLine[2])

        while numEpisode < episode_limit:
            done = False
            total = 0
            state = env.reset()

            while not done:
                if render:
                    env.render()
                
                # stochastic action using the deterministic policy and Gaussian noise
                action = agent.getNoisyAction(state, actionSigma)
                nextState, reward, env_done, info = env.step(action)
                # save the SARSD tuple to the experience buffer
                agent.buffer.store(state, action, reward, nextState, env_done)

                step += 1
                shouldUpdatePolicy = step % policyDelay == 0
                # update weights of all networks according to the TD3 algorithm
                agent.update(
                    miniBatchSize, trainingSigma, trainingClip, shouldUpdatePolicy
                )

                done = env_done
                state = nextState
                total += reward

            # keep track of running average reward
            runningReward = total\
                            if runningReward is None\
                            else runningReward * 0.99 + total * 0.01

            # history logging
            fields = [numEpisode, total, runningReward]
            with open(env.name + '-data.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)

            print(
                    f"episode {numEpisode:6d} --- " +
                    f"total reward: {total:7.2f} --- " +
                    f"running average: {runningReward:7.2f}",
                    flush=True
                )

            agent.save()

            if numEpisode % period == 0 and periodicSaving:
                agent.save(suffix = "_" + str(numEpisode))

            numEpisode += 1

def evaluate_model(index, env_suffix = "", trials = 3):
    """
    Test the learned agent. Actions are drawn using the learned deterministic policy!
    Parameters:
        index - index of the trial in which the agent was trained
        env_suffix - value of the suffix parameter that was passed during the training process
                     for a particular agent
        trials - number of episodes to generate
    """

    env = gym.make(envName)
    env.name = "{}{}_{}".format("BipedalWalker-v3", env_suffix, str(index))
    agent = Agent(env, learningRate, gamma, tau)

    for i in range(trials):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            steps += 1
            env.render()
            action = agent.getDeterministicAction(state)
            nextState, reward, env_done, info = env.step(action)
            state = nextState
            done = env_done

# train(64, periodicSaving = True, period = 100)
evaluate_model(0)