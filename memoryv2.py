import torch
from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state','action','next_state','reward'))
TransitionIdx = namedtuple('transitionIdx', ('idx', 'action', 'reward', 'done'))

USE_CUDA = torch.cuda.is_available()
tType = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class ReplayMemory2(object):
	def __init__(self, capacity, num_history_frames = 4):
		self.capacity = capacity
		self.memory = []
		self.memoryTransitions = []
		self.num_frames = 0
		self.memory_full = False
		self.num_transitions = 0
		self.num_history = num_history_frames

	def getCurrentIndex(self):
		return (self.num_frames-1)%self.capacity

	def pushTransition(self,*args):
		if len(self.memoryTransitions) < self.capacity-1:
			self.memoryTransitions.append(None)
		self.memoryTransitions[self.num_transitions] = TransitionIdx(*args)
		self.num_transitions = (self.num_transitions+1)% (self.capacity-1)

	def pushFrame(self, frame):
		if len(self.memory)< self.capacity:
			self.memory.append(None)
		else:
			self.memory_full = True
		self.memory[self.num_frames] = frame
		self.num_frames = (self.num_frames +1)% self.capacity

	def sampleTransition(self, batch_size):
		rnd_transitions = random.sample(self.memoryTransitions, batch_size)
		output = []
		for i in range(len(rnd_transitions)):
			state = self.memory[rnd_transitions[i][0]]
			for j in range(self.num_history-1):
				idx = rnd_transitions[i][0]-1-j
				if not self.memory_full:
					idx = max(0, idx)
				state = torch.cat((self.memory[(idx)%self.capacity], state),1)

			action = rnd_transitions[i][1]
			reward = rnd_transitions[i][2]
			output.append(None)
			if rnd_transitions[i][3]:
				output[i] = Transition(state.type(tType)/255.0, action, None, reward)
			else:
				next_state = self.memory[(rnd_transitions[i][0]+1)%self.capacity]
				for j in range(self.num_history-1):
					idx =  rnd_transitions[i][0]-j
					if not self.memory_full:
						idx = max(0, idx)
					next_state = torch.cat((self.memory[(idx)%self.capacity], next_state),1)
				output[i] = Transition(state.type(tType)/255.0, action, next_state.type(tType)/255.0, reward)
		return Transition(*zip(* output))

	def __len__(self):
		return len(self.memory)
