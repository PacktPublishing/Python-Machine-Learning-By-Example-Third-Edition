'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 14 Making Decision in Complex Environments with Reinforcement Learning
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

import torch
import gym


env = gym.make('Blackjack-v0')

env.reset()

env.step(1)

env.step(1)

env.step(0)



def run_episode(env, hold_score):
    state = env.reset()
    rewards = []
    states = [state]
    while True:
        action = 1 if state[0] < hold_score else 0
        state, reward, is_done, info = env.step(action)
        states.append(state)
        rewards.append(reward)
        if is_done:
            break
    return states, rewards



from collections import defaultdict

def mc_prediction_first_visit(env, hold_score, gamma, n_episode):
    V = defaultdict(float)
    N = defaultdict(int)
    for episode in range(n_episode):
        states_t, rewards_t = run_episode(env, hold_score)
        return_t = 0
        G = {}
        for state_t, reward_t in zip(states_t[1::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[state_t] = return_t
        for state, return_t in G.items():
            if state[0] <= 21:
                V[state] += return_t
                N[state] += 1
    for state in V:
        V[state] = V[state] / N[state]
    return V


gamma = 1
hold_score = 18
n_episode = 500000


value = mc_prediction_first_visit(env, hold_score, gamma, n_episode)

print(value)

print('Number of states:', len(value))




