#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 18:53:57 2017

@author: max
"""
import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from random import random
from collections import deque

from environment import Environment
from dqn import DQN
from replaymemory import ReplayMemory

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 100000

class Agent(object):
    def __init__(self,
                 game,
                 dimensions,
                 mem_size = 10000,
                 observation_buffer_size = 4,
                 learning_rate = 1e-4,
                 downsampling_rate=2,
                 record=False, 
                 seed=0):
        """
        Inputs:
        - game: string to select the game
        - dimensions: tuple (h1,h2,w1,w2) with dimensions of the game (to crop borders)
                    breakout: (32, 195, 8, 152)
        - mem_size: int length of the replay memory
        - observation_buffer_size: int number of recent frames used as input for neural network
        - learning_rate: float
        - downsampling_rate: int 
        - record: boolean to enable record option
        - seed: int to reproduce results
        """
        
        # Cuda
        self.use_cuda = torch.cuda.is_available()
        
        # Environment
        self.env = Environment(game, dimensions, downsampling_rate, record, seed)
        
        # Neural network
        self.net = DQN(channels_in = observation_buffer_size,
                       num_actions = self.env.get_number_of_actions())
        if self.use_cuda:
            self.net.cuda()
            
        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        # self.optimizer = optim.RMSprop(model.parameters(), lr = learning_rate,alpha=alpha, eps=epsilon)
        
        # Replay Memory
        self.replay = ReplayMemory(mem_size)
        
        # Buffer for the most recent observations
        self.observation_buffer_size = observation_buffer_size
        self.observation_buffer = deque(maxlen=observation_buffer_size)
        # Initialize observation buffer
        initial_observation = self.env.get_observation()
        self.init_observations(initial_observation)
        
        # Epsilon
        self.epsilon = EPSILON_START
        
        # Steps
        self.step = 0
        
    def init_observations(self, observation):
        """
        Initialize the observation buffer
        
        Inputs:
        - observation: observation to initialize the buffer
        
        Returns:
        - observations: np.array with the initalized buffer
        """
        for _ in range(self.observation_buffer_size):
            self.add_observation(observation)
        
        return np.array(self.observation_buffer)
        
    def add_observation(self, observation):
        """
        Adds a observation to the internal observation buffer
        
        Inputs:
        - observation: observation to append
        """
        self.observation_buffer.append(observation)
        
    def get_recent_observations(self):
        """
        Returns the most recent observations
        
        Returns:
        - observations: np.array with the most recent observations
        """
        return np.array(self.observation_buffer)
    
    def reshape_observations(self, observation):
        """
        Reshape observation to fit into PyTorch framework
        
        Inputs:
        - observations: np.array
        
        Returns:
        - observations: np.array
        """
        return observation.reshape(1, # 1 sample
                             self.observation_buffer_size, # observations
                             self.env.get_width(), # width of observation
                             self.env.get_height() # height of observation
                             )
        
    def select_action(self, observations):
        """
        Select an random action from action space or an proposed action
        from neural network depending on epsilon
        
        Inputs:
        - observations: np.array with observations
        
        Returns:
        action: int
        """
        # Decrease epsilon value
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                                    np.exp(-1. * self.step / EPSILON_DECAY)
                                    
        if self.epsilon > random():
            print('random action')
            # Random action
            action = self.env.sample_action()
            # TODO: Check tensor types
            action = torch.LongTensor([[action]])
        else:
            print('dqn action')
            # Action according to neural net
            observations = self.reshape_observations(observations)
            # Wrap tensor into variable
            if self.use_cuda:
                observations_variable = Variable(torch.FloatTensor(observations).cuda())
            else:
                observations_variable = Variable(torch.FloatTensor(observations))
            # Evaluate network and return action with maximum of activation
            action = self.net(observations_variable).data.cpu().max(1)[1].view(1,1)
            
        return action
    
    def play(self):
        """
        """
        done = False
        
        # Reset game
        obs = self.env.reset()
        observations = self.init_observations(obs)
        
        while not done:
            observations = self.reshape_observations(observations)
            
            # Wrap tensor into variable
            if self.use_cuda:
                observations_variable = Variable(torch.FloatTensor(observations).cuda())
            else:
                observations_variable = Variable(torch.FloatTensor(observations))
            
            # Evaluate network and return action with maximum of activation
            action = self.net(observations_variable).data.cpu().max(1)[1].view(1,1)
            
            # TODO: make a nice wrapper
            # Render game
            self.env.game.render(mode='human')
            
            observation, reward, done, info = self.env.step(action)
            
            self.add_observation(observation)
            
            observations = self.get_recent_observations()
            print('Action', action, 'reward', reward)
            
        self.env.game.close()