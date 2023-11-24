'''
SARDINE
Copyright (c) 2023-present NAVER Corp. 
MIT license
'''

import gymnasium as gym
import numpy as np
import sardine
from sardine.wrappers import IdealState
from sardine.policies import EpsilonGreedyOracle, EpsilonGreedyAntiOracle

np.set_printoptions(precision=3, suppress=True)

## Let's create the environment of our choice
env = gym.make("SlateRerank-Static-v0")

## If you want to work with Fully observable state, add a wrapper to the environment
env = IdealState(env)

## Generate a dataset of 10 users with 50% random actions and 50% greedy actions
logging_policy = EpsilonGreedyOracle(epsilon = 0.0, env = env, seed = 2023)
dataset = env.generate_dataset(n_users = 10, policy = logging_policy, seed = 2023, dataset_type="dict")

## After training your agent on the dataset, evaluate in the simulator. Here an example with a random agent.
observation, info = env.reset(seed = 2024)
env.action_space.seed(2024)
cum_reward_list, cum_boredom_list = [], []
cum_reward, cum_boredom = 0, 0
ep = 0
while ep < 100:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    clicks = info["clicks"]
    slate = info["slate"]

    cum_reward += reward
    cum_boredom += (1.0 if np.sum(info["bored"] == True) > 0 else 0.0)

    if terminated or truncated:
        cum_reward_list.append(cum_reward)
        cum_boredom_list.append(cum_boredom)
        cum_reward = 0
        cum_boredom = 0
        observation, info = env.reset()
        ep += 1

print("Average return: ", np.mean(cum_reward_list))

env.close()