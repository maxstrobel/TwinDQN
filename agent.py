#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
from random import random

from environment import Environment
from dqn import DQN
from replaymemory import ReplayMemory
from collections import namedtuple

# if gpu is to be used
use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

Transition = namedtuple('Transition', ('state','action','next_state','reward'))

def gray2pytorch(img):
    return torch.from_numpy(img[:,:,None].transpose(2, 0, 1)).unsqueeze(0)

class Agent(object):
    def __init__(self,
                 game,
                 mem_size = 1024*512, # one element needs around 30kB => 100k == 3 GB
                 state_buffer_size = 4,
                 batch_size = 64,
                 learning_rate = 1e-4,
                 pretrained_model = None,
                 record=False,
                 seed=0):
        """
        Inputs:
        - game: string to select the game
        - mem_size: int length of the replay memory
        - state_buffer_size: int number of recent frames used as input for neural network
        - batch_size: int
        - learning_rate: float
        - pretrained_model: str path to the model
        - record: boolean to enable record option
        - seed: int to reproduce results
        """

        # Namestring
        self.game = game

        # dimensions: tuple (h1,h2,w1,w2) with dimensions of the game (to crop borders)
        if self.game == 'Breakout-v0':
            dimensions = (32, 195, 8, 152)
        elif self.game == 'SpaceInvaders-v0':
            dimensions = (21, 195, 20, 141)

        # Environment
        self.env = Environment(game, dimensions)

        # Cuda
        self.use_cuda = torch.cuda.is_available()

        # Neural network
        self.net = DQN(channels_in = state_buffer_size,
                       num_actions = self.env.get_number_of_actions())

        self.target_net = DQN(channels_in = state_buffer_size,
                       num_actions = self.env.get_number_of_actions())
        if self.use_cuda:
            self.net.cuda()
            self.target_net.cuda()

        if pretrained_model:
            self.net.load(pretrained_model)
            self.target_net.load(pretrained_model)
            self.pretrained_model = True
        else:
            self.pretrained_model = False

        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        # self.optimizer = optim.RMSprop(self.net.parameters(), lr = learning_rate,alpha=alpha, eps=epsilon)

        # Batch size - optimization
        self.batch_size = batch_size

        # Fill replay memory before training
        if not self.pretrained_model:
            self.start_train_after = 50000
        else:
            self.start_train_after = mem_size//2

        # Optimize each k frames
        self.optimize_each_k = 4

        # Updates for target_net
        self.update_target_net_each_k_steps = 10000

        # Save
        self.save_net_each_k_episodes = 500

        # Frame skips
        self.frame_skips = 4

        # Replay Memory (Long term memory)
        #self.replay = ReplayMemory(mem_size)
        self.replay = ReplayMemory(capacity=mem_size, num_history_frames=state_buffer_size)

        # Buffer for the most recent states (Short term memory)
        self.num_stored_frames = state_buffer_size

        # Steps
        self.steps = 0


    def select_action(self, observation):
        """
        Select an random action from action space or an proposed action
        from neural network depending on epsilon

        Inputs:
        - observation: np.array with the observation

        Returns:
        action: int
        """
        # Hyperparameters
        EPSILON_START = 0.9
        EPSILON_END = 0.01
        EPSILON_DECAY = 75000

        # Decrease of epsilon value
        if not self.pretrained_model:
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                                    np.exp(-1. * (self.steps-self.batch_size) / EPSILON_DECAY)
        else:
            epsilon = EPSILON_END

        if epsilon > random():
            # Random action
            action = self.env.sample_action()
            action = LongTensor([[action]])
        else:
            # Action according to neural net
            # Wrap tensor into variable
            state_variable = Variable(observation, volatile=True)

            # Evaluate network and return action with maximum of activation
            action = self.net(state_variable).data.max(1)[1].view(1,1)

        return action


    def optimize(self, net_updates):
        """
        Optimizer function

        Inputs:
        - net_updates: int
        """
        # Hyperparameter
        GAMMA = 0.99

        #   not enough memory yet
        if len(self.replay) < self.start_train_after:
            return

        # Sample a transition
        batch = self.replay.sampleTransition(self.batch_size)

        # Mask to indicate which states are not final (=done=game over)
        non_final_mask = ByteTensor(list(map(lambda ns: ns is not None, batch.next_state)))

        # Wrap tensors in variables
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        non_final_next_states = Variable(torch.cat([ns for ns in batch.next_state if ns is not None]),
                                              volatile=True) # volatile==true prevents calculation of the derivative

        next_state_values = Variable(torch.zeros(self.batch_size).type(FloatTensor), volatile=False)

        if self.use_cuda:
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            non_final_mask = non_final_mask.cuda()
            non_final_next_states = non_final_next_states.cuda()
            next_state_values = next_state_values.cuda()

        # Compute Q(s_t, a) - the self.net computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_max_values = self.target_net(non_final_next_states).detach().max(1)[0]
        next_state_values[non_final_mask]= next_max_values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        bellman_error = expected_state_action_values - state_action_values
            # clip the bellman error between [-1 , 1]
        clipped_bellman_error = bellman_error.clamp(-1, 1)
            # Note: clipped_bellman_delta * -1 will be right gradient
        d_error = clipped_bellman_error * -1.0
            # Clear previous gradients before backward pass
        self.optimizer.zero_grad()
            # run backward pass
        state_action_values.backward(d_error.data.unsqueeze(1))


        #loss.backward()
        #for param in self.net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if (net_updates%self.update_target_net_each_k_steps)==0 and net_updates!=0:
            self.target_net.load_state_dict(self.net.state_dict())
            print('target_net update!')


    def play(self):
        """
        Play a game with the current net and render it
        """
        done = False # games end indicator variable
        score = 0
        # Reset game
        observation = self.env.reset()
        state = self.init_state(observation)

        while not done:
            action = self.select_action(state)

            # Render game
            self.env.game.render(mode='human')

            observation, reward, done, info = self.env.step(action)
            score += reward
            self.add_observation(observation)

            state = self.get_recent_state()

        print('Final score:', score)
        self.env.game.close()


    def train(self):
        """
        Train the agent
        """
        num_episodes = 100000
        net_updates = 0

        # Logging
        log_avg_episodes = 100
        best_score = 0
        avg_score = 0
        filename = self.game + '_train.log'

        open(filename, 'w').close() # empty file

        print('Started training...')

        for i_episode in range(num_episodes):
            # reset game at the start of each episode
            screen = self.env.reset()

            # list of k last frames
            last_k_frames = []
            for j in range(self.num_stored_frames):
                last_k_frames.append(None)
                last_k_frames[j] = gray2pytorch(screen)

            if i_episode == 0:
                self.replay.pushFrame(last_k_frames[0].cpu())

            # frame is saved as ByteTensor -> convert to gray value between 0 and 1
            state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0

            done = False # games end indicator variable
            total_reward = 0 # reset score

            # Loop over one game
            while not done:
                self.steps +=1

                action = self.select_action(state)

                # perform selected action on game
                screen, reward, done, info = self.env.step(action[0,0])#envTest.step(action[0,0])

                #   clamp rewards
                reward = torch.Tensor([np.clip(reward,-1,1)])
                total_reward += reward[0]

                #   save latest frame, discard oldest
                for j in range(self.num_stored_frames-1):
                    last_k_frames[j] = last_k_frames[j+1]
                last_k_frames[self.num_stored_frames-1] = gray2pytorch(screen)

                # convert frames to range 0 to 1 again
                if not done:
                    next_state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0
                else:
                    next_state = None

                # Store transition
                self.replay.pushFrame(last_k_frames[self.num_stored_frames - 1].cpu())
                self.replay.pushTransition((self.replay.getCurrentIndex()-1)%self.replay.capacity, action, reward, done)

                #	only optimize each kth step
                if self.steps%self.optimize_each_k == 0:
                    self.optimize(net_updates)
                    net_updates += 1

                # set current state to next state to select next action
                if next_state is not None:
                    state = next_state

                if self.use_cuda:
                    state = state.cuda()

                # plays episode until there are no more lives left ( == done)
                if done:
                    break;

            # TODO: better logging...
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

            if i_episode % self.save_net_each_k_episodes == 0:
                self.target_net.save('modelParams/' + self.game + '-' + str(i_episode) + '_episodes')

        print('Training done!')
        self.target_net.save('modelParams/' + self.game + '.model')
