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
import gym

import numpy as np
import math
import os
from itertools import count
from random import random, randrange
from collections import deque

from environment import Environment
from dqn import DQN
from replaymemory import ReplayMemory
from memoryv2 import ReplayMemory2
from collections import namedtuple
from utils import gray2pytorch, breakout_preprocess


EPSILON_START = 0.95
EPSILON_END = 0.01
EPSILON_DECAY = 50000

# if gpu is to be used
use_cuda = torch.cuda.is_available()

path_to_dir = os.getcwd()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

Transition = namedtuple('Transition', ('state','action','next_state','reward'))

class Agent(object):
    def __init__(self,
                 game,
                 dimensions,
                 mem_size = 100000, # one element needs around 30kB => 100k == 3 GB
                 state_buffer_size = 4,
                 learning_rate = 1e-4,
                 preload_model = False,
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

        self.target_net = DQN(channels_in = state_buffer_size,
                       num_actions = self.env.get_number_of_actions())
        if self.use_cuda:
            self.net.cuda()
            self.target_net.cuda()


        self.preload_model = preload_model
        if preload_model:
            # TODO: implement load functionality
            pass

        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        # self.optimizer = optim.RMSprop(self.net.parameters(), lr = learning_rate,alpha=alpha, eps=epsilon)

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

        # Updates for target_net
        self.update_target_net_each_k = 10000

        # Save
        self.save_net_each_k = 1000

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
        if self.preload_model:
            self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                                    np.exp(-1. * (self.steps-self.batch_size) / EPSILON_DECAY)
        else:
            self.epsilon = EPSILON_END

        if self.epsilon > random():
            # Random action
            action = self.env.sample_action()
            action = LongTensor([[action]])
        else:
            # Action according to neural net
            # Wrap tensor into variable
            state_variable = Variable(FloatTensor(state[None,:,:,:].astype(float)/255.0), volatile=True)

            # Evaluate network and return action with maximum of activation
            action = self.net(state_variable).data.max(1)[1].view(1,1)

        return action

    def train(self):
        """

        """
        log_avg_episodes = 100
        num_episodes = 100000
        best_score = 0
        avg_score = 0
        net_updates = 0

        # TODO: timestamp for log file
        filename = 'dqn_train.log'

        open(filename, 'w').close() # empty file

        # Loop over games to play
        for i_episode in range(1, num_episodes):
            # Reset environment
            obs = self.env.reset()
            state = self.init_state(obs)
            self.replay.init_memory(state)
            done = False # games end indicator variable
            total_reward = 0 # reset score

            # Loop over one game
            while not done:
                #self.env.game.render() # TODO: comment out => only debug
                action = self.select_action(state)

                # skip some frames
                for _ in range(self.frame_skips):
                    observation, reward, done, info = self.env.step(action[0,0])
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
                self.replay.push(action.cpu().numpy(), reward, next_state)

                # Update state
                state = next_state

                # Check if samples are enough to optimize
                if len(self.replay) >= self.batch_size:
                    self.optimize(net_updates)
                    net_updates += 1

                self.steps += 1

            # TODO: better loggging...
            print('Episode', i_episode,
                  '\treward', total_reward,
                  '\tnum_steps', self.steps,
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

            if i_episode % self.save_net_each_k == 0:
                torch.save(self.net.state_dict(),path_to_dir+'./modelParams/dqn_' + str(i_episode) + '_episodes')

        print('Training done!')
        torch.save(self.net.state_dict(),path_to_dir+'./modelParams/dqn_final')

    def optimize(self, net_updates):
        """
        """
        gamma = 0.99 # nature paper

        # Sample a transition
        batch = self.replay.sample(self.batch_size)

        # Mask to indicate which states are not final (=done=game over)
        non_final_mask = ByteTensor(list(map(lambda ns: ns is not None, batch.next_state)))

        # Wrap tensors in variables
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        non_final_next_states = Variable(torch.cat([ns for ns in batch.next_state if ns is not None]),
                                              volatile=True) # volatile==true prevents calculation of the derivative

        next_state_values = Variable(torch.zeros(self.batch_size).type(FloatTensor), volatile=False)

        # Compute Q(s_t, a) - the self.net computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_max_values = self.target_net(non_final_next_states).detach().max(1)[0]
        next_state_values[non_final_mask]= next_max_values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch
        # Detach unused states from computation
        #expected_state_action_values[final_mask] = expected_state_action_values[final_mask].detach()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the self.net
        self.optimizer.zero_grad()

        # loss = expected_state_action_values - state_action_values
        # loss = loss.clamp(-1.0,1.0) * -1.0
        # state_action_values.backward(loss.data.unsqueeze(1))
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)


        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if net_updates % self.update_target_net_each_k == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            print('target_net update!')


        # reward_score = int(torch.sum(reward_batch).data.cpu().numpy()[0])

        # return loss.data.cpu().numpy(), reward_score




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
 
    def dqn_learning(self,
        num_frames = 4,
        batch_size = 64,
        mem_size = 1048576,
        start_train_after = 10000,
        num_episodes = 10000,
        update_params_each_k = 10000,
        optimize_each_k = 4,
        train = True,
        preload_model = False,
        game = "Breakout-v0"):
        #   num_frames: history of frames as input to DQN
        #   batch_size: size of random samples of memory
        env = gym.make(game)

        num_actions = env.action_space.n
        if not train:
            preload_model = True
			
        if preload_model:
            self.net.load_state_dict(torch.load(path_to_dir+'\modelParams\paramsWithTargetAfter4700'+game))
            self.target_net.load_state_dict(torch.load(path_to_dir+'\modelParams\paramsWithTargetAfter4700'+game))
            
            # to fill memory first 
            start_train_after = mem_size
        #initialize optimizer
        memory = ReplayMemory2(mem_size, num_history_frames = num_frames)

        num_param_updates = 0

        #   greedy_epsilon_selection of an action
        def select_action(dqn, observation,eps):
            rnd = random()
            if rnd < eps:
                return torch.LongTensor([[randrange(num_actions)]])
            else:
                return dqn(Variable(observation, volatile=True)).type(torch.FloatTensor).data.max(1)[1].view(1,1)

        #   function to optimize self.net according to reinforcement_q_learning.py's optimization function
        def optimization(last_state, num_param_updates):
            #   not enough memory yet
            if len(memory) < start_train_after:
                return
            #   get random samples
            batch = memory.sampleTransition(batch_size)

            #   mask of which states are not final states(done = True from env.step)
            non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

            non_final_next_states = Variable(torch.cat(
                        [ns for ns in batch.next_state if ns is not None]),volatile=True)
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.cat(batch.action))
            reward_batch = Variable(torch.cat(batch.reward))

            if use_cuda:
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                non_final_mask = non_final_mask.cuda()
                non_final_next_states = non_final_next_states.cuda()

            state_action_values = torch.gather(self.net(state_batch), 1, action_batch)

            next_state_values = Variable(torch.zeros(batch_size).type(FloatTensor))
            
            next_max_values = self.target_net(non_final_next_states).detach().max(1)[0]
            next_state_values[non_final_mask]= next_max_values

            #next_state_values[non_final_mask] = self.net(non_final_next_states).max(1)[0]

            next_state_values.volatile = False
            expected_state_action_values = (next_state_values*0.99) + reward_batch

            self.optimizer.zero_grad()

            #loss = expected_state_action_values - state_action_values
            #loss = loss.clamp(-1.0,1.0) * -1.0
            #state_action_values.backward(loss.data.unsqueeze(1))

            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            loss.backward()
            for param in self.net.parameters():
            	param.grad.data.clamp_(-1,1)

            self.optimizer.step()

            if num_param_updates % update_params_each_k  == 0:
                self.target_net.load_state_dict(self.net.state_dict())
                print('target_net update!')

        episodes = num_episodes

        num_steps = 0
        avg_score = 0
        best_score = 0
        #torch.save(self.net.state_dict(),path_to_dir+'\modelParams\paramsStart'+game)
        eps_decay = 50000
        for i in range(episodes):
            env.reset()
            screen = env.render(mode='rgb_array')
            #obsTest = envTest.reset()
            # # list of k last frames
            last_k_frames = []
            for j in range(num_frames):
                last_k_frames.append(None)
                last_k_frames[j] = gray2pytorch(breakout_preprocess(screen))#rgb2gr(get_screen_resize(env))
            if i == 0:
                memory.pushFrame(last_k_frames[0].cpu())
            #last_k_frames = np.squeeze(last_k_frames, axis=1)
            state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0

            total_reward = 0
            current_lives = 5
            last_lives = 5
            for t in count():
                # epsilon for greedy epsilon selection, with epsilon decay
                action = torch.LongTensor([[1]])
                
                if current_lives == last_lives:
                    if not preload_model:
                        eps = 0.01 + (0.95-0.01)*math.exp(-1.*(num_steps-start_train_after)/eps_decay)
                        action = select_action(self.net, state, eps)
                    else:
                        action = select_action(self.net, state, 0.01)                    
                else:
                    current_lives = last_lives
                    
                num_steps +=1
                _, reward, done, info = env.step(action[0,0])#envTest.step(action[0,0])
                last_lives = info['ale.lives']
                
                    #reward = -1.0
                #   clamp rewards
                reward = torch.Tensor([max(-1.0,min(reward,1.0))])
                total_reward += reward[0]

                #   save latest frame, discard oldest
                screen = env.render(mode='rgb_array')
                for j in range(num_frames-1):
                    last_k_frames[j] = last_k_frames[j+1]
                last_k_frames[num_frames-1] = gray2pytorch(breakout_preprocess(screen))#torch.from_numpy(envTest.get_observation())#rgb2gr(get_screen_resize(env))

                if not done:
                    next_state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0
                else:
                    next_state = None

                #   save to memory
                if train:
                    memory.pushFrame(last_k_frames[num_frames - 1].cpu())
                    memory.pushTransition((memory.getCurrentIndex()-1)%memory.capacity, action, reward, done)
                    
                    if num_steps % optimize_each_k==0:
                        optimization(state,num_param_updates)
                        num_param_updates+=1

                if next_state is not None and use_cuda:
                    state = next_state.cuda()

                if done:
                    break;
                if not train:
                    env.render()
            avg_score += total_reward
            print("episode: ",(i+1),"\treward: ",total_reward, "\tnum steps: ", num_steps)
            if total_reward > best_score:
                best_score = total_reward
            if (i-49) % 50 == 0:
                print("For 50 episodes:\taverage score: ", avg_score/50, "\tbest score so far: ", best_score)
                avg_score = 0
            if (i-200) % 500 == 0:
                        torch.save(self.net.state_dict(),path_to_dir+'/modelParams/paramsWithTargetAfter'+str(i)+game)
        torch.save(self.net.state_dict(),path_to_dir+'/modelParams/paramsWithTargetFinal'+game)
