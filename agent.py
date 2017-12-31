#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 18:53:57 2017

@author: max
"""
import torch
import torch.optim as optim
from torch.nn import functional as F
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

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

class Agent(object):
    def __init__(self,
                 game,
                 dimensions,
                 mem_size = 100000, # one element needs around 60kB => 100k == 6 GB
                 state_buffer_size = 4,
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
        - state_buffer_size: int number of recent frames used as input for neural network
        - learning_rate: float
        - record: boolean to enable record option
        - seed: int to reproduce results
        """
        
        # Cuda
        self.use_cuda = torch.cuda.is_available()

        # Environment
        self.env = Environment(game, dimensions)
        
        # Neural network
        self.net = DQN(channels_in = state_buffer_size,
                       num_actions = self.env.get_number_of_actions())
        if self.use_cuda:
            self.net.cuda()
            
        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        # self.optimizer = optim.RMSprop(model.parameters(), lr = learning_rate,alpha=alpha, eps=epsilon)
        
        # Replay Memory (Long term memory)
        self.replay = ReplayMemory(mem_size)
        
        # Buffer for the most recent states (Short term memory)
        self.state_buffer_size = state_buffer_size
        self.state_buffer = deque(maxlen=state_buffer_size)
        # Initialize state buffer
        initial_observation = self.env.get_observation()
        self.init_state(initial_observation)
        
        # Epsilon
        self.epsilon = EPSILON_START
        
        # Batch size - optimization
        self.batch_size = 32
        
        # Steps
        self.steps = 0
        
        # Frame skips
        self.frame_skips = 4 # nature paper
        
        
    def init_state(self, observation):
        """
        Initialize the state buffer
        
        Inputs:
        - observation: observation to initialize the buffer
        
        Returns:
        - state: np.array with the initalized buffer
        """
        for _ in range(self.state_buffer_size):
            self.add_observation(observation)
        
        return np.array(self.state_buffer)
        
    def add_observation(self, observation):
        """
        Adds a observation to the internal state buffer
        
        Inputs:
        - observation: observation to append
        """
        self.state_buffer.append(observation)
        
    def get_recent_state(self):
        """
        Returns the most recent state
        
        Returns:
        - state: np.array with the most recent state
        """
        return np.array(self.state_buffer)
        
    def select_action(self, state):
        """
        Select an random action from action space or an proposed action
        from neural network depending on epsilon
        
        Inputs:
        - state: np.array with the state
        
        Returns:
        action: int
        """
        # TODO: make self.epsilon local
        # Decrease epsilon value
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                                    np.exp(-1. * self.steps / EPSILON_DECAY)
                                    
        if self.epsilon > random():
            # Random action
            action = self.env.sample_action()
            action = LongTensor([[action]])
        else:
            # Action according to neural net
            # Wrap tensor into variable
            state_variable = Variable(FloatTensor(state[None,:,:,:]))

            # Evaluate network and return action with maximum of activation
            action = self.net(state_variable).data.cpu().max(1)[1].view(1,1)

        return action
    
    def train(self):
        """
        
        """
        save_model_episodes = 10000
        log_avg_episodes = 100
        num_episodes = 100000
        best_score = 0
        avg_score = 0
        
        # TODO: timestamp for log file
        filename = 'dqn_train.log' 
    
        open(filename, 'w').close() # empty file
    
        # Loop over games to play
        for i_episode in range(1, num_episodes):
            # Reset environment
            obs = self.env.reset()
            state = self.init_state(obs)
            done = False # games end indicator variable
            total_reward = 0 # reset score
            # Loop over one game
            while not done:
                #self.env.game.render() # TODO: comment out => only debug
                action = self.select_action(state)
                
                # skip some frames
                for _ in range(self.frame_skips):
                    observation, reward, done, info = self.env.step(action)
                    # Add current observation to state buffer (short term memory)
                    self.add_observation(observation)
                    total_reward += reward
                
                    # Exit frame skipping loop, if game over
                    if done:
                        break
                
                # Update next_state
                if not done:
                    next_state = self.get_recent_state()
                else:
                    next_state = None
                    break
                    
                # Store current transition in replay memory (long term memory)
                
                self.replay.push(state, action.cpu().numpy(), reward, next_state)
                
                # Update state
                state = next_state
                
                # Check if samples are enough to optimize
                if len(self.replay) >= self.batch_size:
                    loss, reward_sum = self.optimize()
                
                self.steps += 1
                
            # TODO: better loggging...
            print('Episode', i_episode,
                  '\tloss', loss,
                  '\treward', total_reward,
                  '\treplay size', len(self.replay))
            
            avg_score += total_reward
            if total_reward > best_score:
                best_score = total_reward
                
            if i_episode % log_avg_episodes == 0:
                print('Episode:', i_episode,
                      'avg on last', log_avg_episodes, 'games:', avg_score/log_avg_episodes,
                      'best score so far:', best_score)
                # Logfile
                with open(filename, "a") as logfile:
                    logfile.write('Episode: '+ str(i_episode)+
                                  '\t\t\tavg on last ' + str(log_avg_episodes) +
                                  ' games: ' + str(avg_score/log_avg_episodes) +
                                  '\t\t\tbest score so far: ' + str(best_score) + '\n')
                avg_score = 0
                
            if i_episode % save_model_episodes == 0:
                self.net.save('dqn_' + str(i_episode) + '_episodes.model')
                
    def optimize(self):
        """
        """
        gamma = 0.99 # nature paper
        
        # Sample a transition
        transition = self.replay.sample(self.batch_size)
        
        # Mask to indicate which states are not final (=done=game over)
        non_final_mask = ByteTensor(list(map(lambda ns: ns is not None, transition.next_state)))
        final_mask = 1 - non_final_mask
        
        # Wrap tensors in variables
        state_batch = Variable(torch.cat(transition.state))
        action_batch = Variable(torch.cat(transition.action))
        reward_batch = Variable(torch.cat(transition.reward))
        non_final_next_state_batch = Variable(torch.cat(
        [ns for ns in transition.next_state if ns is not None]))
        next_state_action_values = Variable(torch.zeros(self.batch_size).type(FloatTensor))
        
        # volatile==true prevents calculation of the derivative 
        non_final_next_state_batch.volatile = True
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        next_state_action_values[non_final_mask] = self.net(non_final_next_state_batch).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_action_values * gamma) + reward_batch
        # Detach unused states from computation
        expected_state_action_values[final_mask] = expected_state_action_values[final_mask].detach()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        
        reward_score = int(torch.sum(reward_batch).data.cpu().numpy()[0])

        return loss.data.cpu().numpy(), reward_score
        
       
       
        
    def play(self):
        """
        """
        done = False # games end indicator variable
        
        # Reset game
        observation = self.env.reset()
        state = self.init_state(observation)
        
        while not done:
            # Wrap tensor into variable
            state_variable = Variable(FloatTensor(state[None,:,:,:]))
            
            # Evaluate network and return action with maximum of activation
            action = self.net(state_variable).data.cpu().max(1)[1].view(1,1)
            
            # TODO: make a nice wrapper
            # Render game
            self.env.game.render(mode='human')
            
            observation, reward, done, info = self.env.step(action)
            
            self.add_observation(observation)
            
            state = self.get_recent_state()
            print('Action', action, 'reward', reward)
            
        self.env.game.close()