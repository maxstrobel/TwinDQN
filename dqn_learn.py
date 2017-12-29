import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import gym
import gym.spaces

import math
import os
import numpy as np
import random
from itertools import count
from collections import namedtuple
from copy import deepcopy

from utils import get_screen_resize, rgb2gr

from dqn import DQN

USE_CUDA = torch.cuda.is_available()
tType = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

Transition = namedtuple('Transition', ('state','action','next_state','reward'))
TransitionIdx = namedtuple('transitionIdx', ('idx', 'action', 'reward', 'done'))

path_to_dir = os.getcwd()

class ReplayMemory(object):
	def __init__(self, capacity, num_history_frames = 4):
		self.capacity = capacity
		self.memory = []
		self.memoryTransitions = []
		self.num_frames = 0
		self.num_transitions = 0
		self.num_history = num_history_frames

	def getCurrentIndex(self):
		return (self.num_frames-1)%self.capacity

	def pushTransition(self,*args):
		if len(self.memoryTransitions) < self.capacity:
			self.memoryTransitions.append(None)
		self.memoryTransitions[self.num_transitions] = TransitionIdx(*args)
		self.num_transitions = (self.num_transitions+1)% self.capacity

	def pushFrame(self, frame):
		if len(self.memory)< self.capacity:
			self.memory.append(None)
		self.memory[self.num_frames] = frame
		self.num_frames = (self.num_frames +1)% self.capacity

	def sampleTransition(self, batch_size):
		rnd_transitions = random.sample(self.memoryTransitions, batch_size)
		output = []
		for i in range(len(rnd_transitions)):
			state = self.memory[rnd_transitions[i][0]]
			next_state = self.memory[rnd_transitions[i][0]+1]
			for j in range(self.num_history-1):
				next_state = torch.cat((self.memory[(rnd_transitions[i][0]-j)%self.capacity],next_state),1)
				state = torch.cat((self.memory[(rnd_transitions[i][0]-1-j)%self.capacity], state),1)

			action = rnd_transitions[i][1]
			reward = rnd_transitions[i][2]
			output.append(None)
			if rnd_transitions[i][3]:
				output[i] = Transition(state.cuda(),action.cuda(), None, reward.cuda())
			else:
				output[i] = Transition(state.cuda(),action.cuda(), next_state.cuda(), reward.cuda())

		return output

	def __len__(self):
		return len(self.memory)


def dqn_learning(
	num_frames = 16,
	batch_size = 128,
	mem_size = 524288,
	learning_rate = 0.00025,
	alpha = 0.95,
	epsilon = 0.01,
	start_train_after = 25000,
	num_episodes = 100000,
	update_params_each_k = 1000,
	optimize_each_k = 5
):
    #   num_frames: history of frames as input to DQN
    #   batch_size: size of random samples of memory
	env = gym.make("Breakout-v0")

	num_actions = env.action_space.n

    #   so far use rgb channels
	model = DQN(channels_in = num_frames,num_actions = num_actions)
	target_model = DQN(channels_in = num_frames, num_actions = num_actions)

	if USE_CUDA:
		model.cuda()
		target_model.cuda()

    #initialize optimizer
	opt = optim.RMSprop(model.parameters(), lr = learning_rate,alpha=alpha, eps=epsilon)
	memory = ReplayMemory(mem_size, num_history_frames = num_frames)

	num_param_updates = 0

    #   greedy_epsilon_selection of an action
	def select_action(dqn, observation,eps):
		rnd = random.random()
		if rnd < eps:
			return torch.LongTensor([[random.randrange(num_actions)]])
		else:
			return dqn(Variable(observation, volatile=True)).type(torch.FloatTensor).data.max(1)[1].view(1,1)

    #   function to optimize model according to reinforcement_q_learning.py's optimization function
	def optimization(last_state, num_param_updates):
        #   not enough memory yet
		if len(memory) < start_train_after:
			return
        #   get random samples
		transitions = memory.sampleTransition(batch_size)
		batch = Transition(*zip(*transitions))

        #   mask of which states are not final states(done = True from env.step)
		non_final_mask = torch.ByteTensor(batch_size)

		for k in range(batch_size):
			if batch.next_state[k] is None:
				non_final_mask[k] = 1
			else:
				non_final_mask[k] = 0

		if batch.next_state[0] is None:
			non_final_next_states = last_state
		else:
			non_final_next_states = batch.next_state[0]

		for x in range(1,batch_size):
			if batch.next_state[x] is None:
				non_final_next_states = torch.cat((non_final_next_states,last_state),0)
			else:
				non_final_next_states = torch.cat((non_final_next_states, batch.next_state[x]),0)
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

		next_max_values = target_model(non_final_next_states).max(1)[0]
		next_state_values = Variable(torch.zeros(batch_size).type(tType))
		next_state_values[non_final_mask]= next_max_values

		#next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

		next_state_values.volatile = False
		expected_state_action_values = (next_state_values*0.999) + reward_batch

		#loss = expected_state_action_values - state_action_values
		#loss = loss.clamp(-1,1) * -1.0

		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

		opt.zero_grad()
		loss.backward()
		#state_action_values.backward(loss.data.unsqueeze(1))
		for param in model.parameters():
			param.grad.data.clamp_(-1,1)
		opt.step()

		if num_param_updates % update_params_each_k  == 0:
			target_model.load_state_dict(model.state_dict())

	episodes = num_episodes

	num_steps = 0
	torch.save(model.state_dict(),path_to_dir+'\modelParams\paramsStart')
	env.reset()
	eps_decay = 75000
	for i in range(episodes):
		env.reset()
        #   list of k last frames
		last_k_frames = []
		for j in range(num_frames):
			last_k_frames.append(None)
			last_k_frames[j] = rgb2gr(get_screen_resize(env))
			memory.pushFrame(last_k_frames[0].cpu())

		state = torch.cat(last_k_frames,1)
		total_reward = 0
		current_lives = 5
		for t in count():
            # epsilon for greedy epsilon selection, with epsilon decay
			eps = 0.05 + (1-0.05)*math.exp(-1.*num_steps/eps_decay)
			action = select_action(model, state, eps)

			_, reward, done, info = env.step(action[0,0])
			lives = info['ale.lives']
			if current_lives != lives:
				current_lives = lives
				#reward = -1.0

			total_reward += reward
            #   clamp rewards
			reward = torch.Tensor([max(-1.0,min(reward,1.0))])

            #   save latest frame, discard oldest
			for j in range(num_frames-1):
				last_k_frames[j] = last_k_frames[j+1]
			last_k_frames[num_frames-1] = rgb2gr(get_screen_resize(env))

			if not done:
				next_state = torch.cat(last_k_frames,1)
			else:
				next_state = None

            #   save to memory

			memory.pushFrame(last_k_frames[num_frames - 1].cpu())
			memory.pushTransition((memory.getCurrentIndex()-1)%memory.capacity, action.cpu(), reward.cpu(), done)
			if num_steps % optimize_each_k==0:
				optimization(state,num_param_updates)
				num_param_updates+=1

			if next_state is not None and USE_CUDA:
				state = next_state.cuda()

			if done:
				break;
			env.render()
		print("episode: ",i,"\treward: ",total_reward, "\tlen of mem: ", len(memory))
		if (i-200) % 250 == 0:
            		torch.save(model.state_dict(),path_to_dir+'\modelParams\paramsAfter'+str(i))
	torch.save(model.state_dict(),path_to_dir+'\modelParams\paramsFinal')
dqn_learning()

