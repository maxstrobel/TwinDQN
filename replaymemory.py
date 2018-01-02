#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 19:50:40 2017

@author: max
"""
import torch
import numpy as np
from collections import deque, namedtuple

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

# State container
State = namedtuple('Transition', ('state', 'action', 'reward'))
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
        self.filling = 0
        self.memory_full = False

    def init_memory(self,state):
        """
        Initalize replay memory with first state
        
        Inputs:
        - state: np.array
        """
        state = State(state[None,:,:,:], None, None)
        self.memory.append(state)
        
    def push(self, action, reward, next_state):
        """
        Pushes the current state into the replay memory
        
        Inputs:
        - action: int
        - reward: int
        - next_state: np.array
        """
        # Update old state with action and reward
        state = self.memory[min(self.filling,self.capacity)].state
        state = State(state, np.uint8(action), np.int8([reward]))
        self.memory[min(self.filling,self.capacity)] = state
        
        # Add next state add the end of the buffer without action and reward
        next_state = State(next_state[None,:,:,:], None, None)
        self.memory.append(next_state)
        self.filling += 1

    def sample(self, batch_size):
        """
        Sample random elements from memory and converts it to tensors
        
        Inputs:
        - batch_size: int
        
        Returns:
        batch: Transistion tensor batch
        """
        # Sample transition
        rand_idx = np.random.randint(min(self.filling,self.capacity), size=batch_size)
        
        # Get samples
        states = [self.memory[idx].state for idx in rand_idx]
        actions = [self.memory[idx].action for idx in rand_idx]
        rewards = [self.memory[idx].reward for idx in rand_idx]
        next_states = [self.memory[idx+1].state for idx in rand_idx]
        
        # Convert to tensor
        states = [FloatTensor(s.astype(float))/255.0 for s in states]
        actions = [LongTensor(a.astype(int).tolist()) for a in actions]
        rewards = [FloatTensor(r.astype(float)) for r in rewards]
        next_states = [FloatTensor(ns.astype(float))/255.0 for ns in next_states]
        
        # Concat tensors
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        
        return Transition(states, actions, rewards, next_states)

    def __len__(self):
        return len(self.memory)