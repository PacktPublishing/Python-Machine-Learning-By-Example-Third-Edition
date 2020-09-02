'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 14 Making Decision in Complex Environments with Reinforcement Learning
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''


import torch

x = torch.empty(3, 4)

print(x)

from gym import envs
print(envs.registry.all())
