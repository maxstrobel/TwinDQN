import gym
import math
import random
from collections import namedtuple
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import numpy as np
import matplotlib

if __name__ =="__main__":
	env = gym.make("SpaceInvaders-v0")
	env.reset()
	env2 = gym.make("Breakout-v0")
	env2.reset()
	for _ in range(1000):
		screen = env.render('rgb_array').transpose()
		env.render()
		env.step(env.action_space.sample())
		env2.render()
		env2.step(env2.action_space.sample())
