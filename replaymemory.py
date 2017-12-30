#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 19:50:40 2017

@author: max
"""
import torch
from collections import deque, namedtuple
from random import sample

# if gpu is to be used
use_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


# Transition container
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity):
        """
        Inputs:
        - capacity: int capacity of the memory
        """
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state):
        """
        Pushes one transition into the replay memory
        
        Inputs:
        - state: np.array
        - action: Tensor
        - reward: int
        - next_state: np.array
        """
        # Create tensors
        state = Tensor(state[None,:,:,:])
        reward = Tensor([reward])
        if next_state is not None:
            next_state = Tensor(next_state[None,:,:,:])
        transition = Transition(state, action, reward, next_state)
        self.memory.append(transition)

    def sample(self, batch_size):
        """
        Sample random elements from memory
        
        Inputs:
        - batch_size: int
        
        Returns:
        transition: Transistion sampled
        """
        transition = sample(self.memory, batch_size)
        return Transition(*(zip(*transition)))

    def __len__(self):
        return len(self.memory)