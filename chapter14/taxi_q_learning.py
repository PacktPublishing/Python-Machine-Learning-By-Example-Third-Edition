'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 14 Making Decision in Complex Environments with Reinforcement Learning
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

import torch
import gym

env = gym.make('Taxi-v3')
n_state = env.observation_space.n
print(n_state)
n_action = env.action_space.n
print(n_action)

env.reset()

env.render()

print(env.step(3))
print(env.step(3))
print(env.step(3))
print(env.step(1))
print(env.step(1))
print(env.step(4))
env.render()
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(5))
env.render()


def gen_epsilon_greedy_policy(n_action, epsilon):
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function


from collections import defaultdict



def q_learning(env, gamma, n_episode, alpha):
    """
    Obtain the optimal policy with off-policy Q-learning method
    @param env: OpenAI Gym environment
    @param gamma: discount factor
    @param n_episode: number of episodes
    @return: the optimal Q-function, and the optimal policy
    """
    n_action = env.action_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        while not is_done:
            action = epsilon_greedy_policy(state, Q)
            next_state, reward, is_done, info = env.step(action)
            delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
            Q[state][action] += alpha * delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            if is_done:
                break
            state = next_state
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy


epsilon = 0.1

epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)


gamma = 1

n_episode = 1000

alpha = 0.4



length_episode = [0] * n_episode
total_reward_episode = [0] * n_episode

optimal_Q, optimal_policy = q_learning(env, gamma, n_episode, alpha)


import matplotlib.pyplot as plt
plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.ylim([-200, 20])
plt.show()

plt.plot(length_episode)
plt.title('Episode length over time')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.show()





