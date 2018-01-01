#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 19:50:40 2017

@author: max
"""
import torch
import numpy as np
from collections import deque, namedtuple
from random import sample

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

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
        - action: int
        - reward: int
        - next_state: np.array
        """
        if next_state is not None:
            next_state = next_state[None,:,:,:]
        transition = Transition(state[None,:,:,:], np.uint8(action), np.int8([reward]), next_state)
        self.memory.append(transition)

    def sample(self, batch_size):
        """
        Sample random elements from memory and converts it to tensors
        
        Inputs:
        - batch_size: int
        
        Returns:
        batch: Transistion tensor batch
        """
        # Sample transition
        transition = sample(self.memory, batch_size)
        transition = Transition(*(zip(*transition)))
        
        # Convert to tensor
        states = transition.state
        actions = transition.action
        rewards = transition.reward
        next_states = transition.next_state
        states = [FloatTensor(s.astype(float))/255.0 for s in states]
        actions = [LongTensor(a.astype(int).tolist()) for a in actions]
        rewards = [FloatTensor(r.astype(float)) for r in rewards]
        next_states = [FloatTensor(ns.astype(float))/255.0 for ns in next_states]
        
        return Transition(states, actions, rewards, next_states)

    def __len__(self):
        return len(self.memory)