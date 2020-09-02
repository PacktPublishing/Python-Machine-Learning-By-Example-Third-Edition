'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 14 Making Decision in Complex Environments with Reinforcement Learning
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

import torch
import gym

env = gym.make('FrozenLake-v0')

gamma = 0.99

threshold = 0.0001


def policy_evaluation(env, policy, gamma, threshold):
    """
    Perform policy evaluation
    @param env: OpenAI Gym environment
    @param policy: policy matrix containing actions and their probability in each state
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the given policy
    """
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(n_state):
            action = policy[state].item()
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                V_temp[state] += trans_prob * (reward + gamma * V[new_state])
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V


def policy_improvement(env, V, gamma):
    """
    Obtain an improved policy based on the values
    @param env: OpenAI Gym environment
    @param V: policy values
    @param gamma: discount factor
    @return: the policy
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.zeros(n_state)
    for state in range(n_state):
        v_actions = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += trans_prob * (reward + gamma * V[new_state])
        policy[state] = torch.argmax(v_actions)
    return policy


def policy_iteration(env, gamma, threshold):
    """
    Solve a given environment with policy iteration algorithm
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: optimal values and the optimal policy for the given environment
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.randint(high=n_action, size=(n_state,)).float()
    while True:
        V = policy_evaluation(env, policy, gamma, threshold)
        policy_improved = policy_improvement(env, V, gamma)
        if torch.equal(policy_improved, policy):
            return V, policy_improved
        policy = policy_improved



V_optimal, optimal_policy = policy_iteration(env, gamma, threshold)
print('Optimal values:\n', V_optimal)
print('Optimal policy:\n', optimal_policy)
