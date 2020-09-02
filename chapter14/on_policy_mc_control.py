'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 14 Making Decision in Complex Environments with Reinforcement Learning
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

import torch
import gym

env = gym.make('Blackjack-v0')



def run_episode(env, Q, n_action):
    """
    Run a episode given Q-values
    @param env: OpenAI Gym environment
    @param Q: Q-values
    @param n_action: action space
    @return: resulting states, actions and rewards for the entire episode
    """
    state = env.reset()
    rewards = []
    actions = []
    states = []
    action = torch.randint(0, n_action, [1]).item()
    while True:
        actions.append(action)
        states.append(state)
        state, reward, is_done, info = env.step(action)
        rewards.append(reward)
        if is_done:
            break
        action = torch.argmax(Q[state]).item()
    return states, actions, rewards



from collections import defaultdict

def mc_control_on_policy(env, gamma, n_episode):
    """
    Obtain the optimal policy with on-policy MC control method
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param n_episode: number of episodes
    @return: the optimal Q-function, and the optimal policy
    """
    n_action = env.action_space.n
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(n_action))
    for episode in range(n_episode):
        states_t, actions_t, rewards_t = run_episode(env, Q, n_action)
        return_t = 0
        G = {}
        for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[(state_t, action_t)] = return_t
        for state_action, return_t in G.items():
            state, action = state_action
            if state[0] <= 21:
                G_sum[state_action] += return_t
                N[state_action] += 1
                Q[state][action] = G_sum[state_action] / N[state_action]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy


gamma = 1

n_episode = 500000

optimal_Q, optimal_policy = mc_control_on_policy(env, gamma, n_episode)

print(optimal_policy)



hold_score = 18

def simulate_episode(env, policy):
    state = env.reset()
    while True:
        action = policy[state]
        state, reward, is_done, _ = env.step(action)
        if is_done:
            return reward


def simulate_hold_episode(env, hold_score):
    state = env.reset()
    while True:
        action = 1 if state[0] < hold_score else 0
        state, reward, is_done, _ = env.step(action)
        if is_done:
            return reward


n_episode = 100000
n_win_opt = 0
n_win_hold = 0

for _ in range(n_episode):
    reward = simulate_episode(env, optimal_policy)
    if reward == 1:
        n_win_opt += 1
    reward = simulate_hold_episode(env, hold_score)
    if reward == 1:
        n_win_hold += 1


print(f'Winning probability:\nUnder the simple policy: {n_win_hold/n_episode}\nUnder the optimal policy: {n_win_opt/n_episode}')
