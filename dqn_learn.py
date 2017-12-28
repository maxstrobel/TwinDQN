import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import gym
import gym.spaces

import math
import numpy as np
import random
from itertools import count
from collections import namedtuple
from copy import deepcopy

from utils import get_screen_resize

from dqn import DQN

USE_CUDA = torch.cuda.is_available()
tType = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

Transition = namedtuple('Transition', ('state','action','next_state','reward'))

class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		if len(self.memory)< self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position+1)%self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


def dqn_learning(
	num_frames = 4,
	batch_size = 2048
):
	env = gym.make("Breakout-v0")

	num_actions = env.action_space.n

	model = DQN(in_channels = 3 * num_frames,num_actions = num_actions)

	if USE_CUDA:
		model.cuda()

	opt = optim.Adam(model.parameters())
	memory = ReplayMemory(200000)


	def select_action(dqn, observation,eps):
		rnd = random.random()
		if rnd > eps:
			return torch.LongTensor([[random.randrange(num_actions)]])
		else:
			return dqn(Variable(observation, volatile=True)).type(torch.FloatTensor).data.max(1)[1].view(1,1)

	def optimization(last_state):
		if len(memory) < batch_size+1:
			return
		transitions = memory.sample(batch_size)
		batch = Transition(*zip(*transitions))

		non_final_mask = torch.LongTensor(tuple(map(lambda s: s is not None, batch.next_state)))

		non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
		for x in range(non_final_next_states.size()[0], batch_size):
			non_final_next_states = torch.cat((non_final_next_states,last_state),0)
		non_final_next_states = Variable(non_final_next_states, volatile=True)
		state_batch = Variable(torch.cat(batch.state))
		action_batch = Variable(torch.cat(batch.action))
		reward_batch = Variable(torch.cat(batch.reward))

		if USE_CUDA:
			state_batch = state_batch.cuda()
			action_batch = action_batch.cuda()
			reward_batch = reward_batch.cuda()
			non_final_mask = non_final_mask.cuda()
			non_final_next_states = non_final_next_states.cuda()

		state_action_values = torch.gather(model(state_batch),1, action_batch)

		next_state_values = Variable(torch.zeros(batch_size).type(tType))

		next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

		next_state_values.volatile = False
		expected_state_action_values = (next_state_values*0.999) + reward_batch

		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

		opt.zero_grad()
		loss.backward()
		for param in model.parameters():
			param.grad.data.clamp_(-1,1)
		opt.step()


	episodes = 1000

	best_mean_episode = -float('inf')
	mean_episode = -float('nan')
	num_steps = 0

	for i in range(episodes):
		env.reset()
		last_k_frames = {}
		for j in range(num_frames):
			last_k_frames[j] = get_screen_resize(env)

		state = torch.cat((last_k_frames[0], last_k_frames[1], last_k_frames[2], last_k_frames[3]),1)
		total_reward = 0

		for t in count():
			eps = 0.01 + (0.9-0.01)*math.exp(-1.*num_steps/200)
			action = select_action(model, state, eps)

			num_steps += 1

			_, reward, done, _ = env.step(action[0,0])
			total_reward += reward
			if reward < 0.0:
				print(reward)
			reward = torch.Tensor([max(-1.0,min(reward,1.0))])

			last_k_frames[0] = last_k_frames[1]
			last_k_frames[1] = last_k_frames[2]
			last_k_frames[2] = last_k_frames[3]
			last_k_frames[3] = get_screen_resize(env)

			if not done:
				next_state = torch.cat((last_k_frames[0], last_k_frames[1], last_k_frames[2], last_k_frames[3]),1)
			else:
				next_state = None

			memory.push(state, action, next_state, reward)

			optimization(state)

			state = next_state

			if done:
				break;
			env.render()
		print(total_reward)

dqn_learning()

