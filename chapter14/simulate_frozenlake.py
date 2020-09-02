'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 14 Making Decision in Complex Environments with Reinforcement Learning
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

import gym
import torch


env = gym.make("FrozenLake-v0")

n_state = env.observation_space.n
print(n_state)
n_action = env.action_space.n
print(n_action)


env.reset()

env.render()

new_state, reward, is_done, info = env.step(2)
env.render()
print(new_state)
print(reward)
print(is_done)
print(info)




def run_episode(env, policy):
    state = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        action = policy[state].item()
        state, reward, is_done, info = env.step(action)
        total_reward += reward
        if is_done:
            break
    return total_reward



n_episode = 1000

total_rewards = []
for episode in range(n_episode):
    random_policy = torch.randint(high=n_action, size=(n_state,))
    total_reward = run_episode(env, random_policy)
    total_rewards.append(total_reward)

print(f'Average total reward under random policy: {sum(total_rewards)/n_episode}')


print(env.env.P[6])
