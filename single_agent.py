#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
from random import random
from collections import namedtuple
from datetime import datetime
import pickle
import os

from environment import Environment
from dqn import DQN
from replaymemory import ReplayMemory

# if gpu is to be used
use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

Transition = namedtuple('Transition', ('state','action','next_state','reward'))

def gray2pytorch(img):
    return torch.from_numpy(img[:,:,None].transpose(2, 0, 1)).unsqueeze(0)

# dimensions: tuple (h1,h2,w1,w2) with dimensions of the game (to crop borders)
dimensions = {'Breakout-v0': (32, 195, 8, 152),
              'SpaceInvaders-v0': (21, 195, 20, 141),
              'Assault-v0': (50, 240, 5, 155),
              'Phoenix-v0': (23, 183, 0, 160),
              'Skiing-v0': (55, 202, 8, 152),
              'Enduro-v0': (50, 154, 8, 160),
              'BeamRider-v0': (32, 180, 9, 159),
              }


class SingleAgent(object):
    def __init__(self,
                 game,
                 mem_size = 800000,
                 state_buffer_size = 4,
                 batch_size = 64,
                 learning_rate = 1e-5,
                 pretrained_model = None,
                 frameskip = 4
                 ):
        """
        Inputs:
        - game: string to select the game
        - mem_size: int length of the replay memory
        - state_buffer_size: int number of recent frames used as input for neural network
        - batch_size: int
        - learning_rate: float
        - pretrained_model: str path to the model
        - record: boolean to enable record option
        """

        # Namestring
        self.game = game

        # Environment
        self.env = Environment(game, dimensions[game], frameskip=frameskip)

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
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(self.net.parameters(), lr=learning_rate,alpha=0.95, eps=0.01)

        self.batch_size = batch_size
        self.optimize_each_k = 1
        self.update_target_net_each_k_steps = 10000
        self.noops_count = 0

        # Replay Memory (Long term memory)
        self.replay = ReplayMemory(capacity=mem_size, num_history_frames=state_buffer_size)
        self.mem_size = mem_size

        # Fill replay memory before training
        if not self.pretrained_model:
            self.start_train_after = 50000
        else:
            self.start_train_after = mem_size//2

        # Buffer for the most recent states (Short term memory)
        self.num_stored_frames = state_buffer_size

        # Steps
        self.steps = 0

        # Save net
        self.save_net_each_k_episodes = 500

    def select_action(self, observation, mode='train'):
        """
        Select an random action from action space or an proposed action
        from neural network depending on epsilon

        Inputs:
        - observation: np.array with the observation

        Returns:
        action: int
        """
        # Hyperparameters
        EPSILON_START = 1
        EPSILON_END = 0.1
        EPSILON_DECAY = 1000000
        MAXNOOPS = 30

        # Decrease of epsilon value
        if not self.pretrained_model:
            #epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            #                        np.exp(-1. * (self.steps-self.batch_size) / EPSILON_DECAY)
            epsilon = EPSILON_START - self.steps * (EPSILON_START - EPSILON_END) / EPSILON_DECAY
        else:
            epsilon = EPSILON_END

        if epsilon < random() or mode=='play':
            # Action according to neural net
            # Wrap tensor into variable
            state_variable = Variable(observation, volatile=True)

            # Evaluate network and return action with maximum of activation
            action = self.net(state_variable).data.max(1)[1].view(1,1)

            # Prevent noops
            if action[0,0]==0:
                self.noops_count += 1
                if self.noops_count == MAXNOOPS:
                    action[0,0] = 1
                    self.noops_count = 0
            else:
                self.noops_count = 0
        else:
            # Random action
            action = self.env.sample_action()
            action = LongTensor([[action]])

        return action


    def optimize(self, net_updates):
        """
        Optimizer function

        Inputs:
        - net_updates: int

        Returns:
        - loss: float
        - q_value: float
        - exp_q_value: float
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
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()


        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if net_updates%self.update_target_net_each_k_steps==0:
            self.target_net.load_state_dict(self.net.state_dict())
            print('target_net update!')

        return loss.data.cpu().numpy()[0]


    def play(self):
        """
        Play a game with the current net and render it
        """
        done = False # games end indicator variable
        score = 0
        # Reset game
        screen = self.env.reset()

        # list of k last frames
        last_k_frames = []
        for j in range(self.num_stored_frames):
            last_k_frames.append(None)
            last_k_frames[j] = gray2pytorch(screen)

        # frame is saved as ByteTensor -> convert to gray value between 0 and 1
        state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0

        while not done:
            action = self.select_action(state, mode='play')

            screen, reward, done, info = self.env.step(action[0,0], mode='play')
            score += reward

            #   save latest frame, discard oldest
            for j in range(self.num_stored_frames-1):
                last_k_frames[j] = last_k_frames[j+1]
            last_k_frames[self.num_stored_frames-1] = gray2pytorch(screen)

            # convert frames to range 0 to 1 again
            state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0

        print('Final score:', score)
        self.env.game.close()


    def train(self):
        """
        Train the agent
        """
        num_episodes = 100000
        net_updates = 0

        # Logging
        sub_dir = self.game + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '/'
        os.makedirs(sub_dir)
        logfile = sub_dir + self.game + '_train.log'
        loss_file = sub_dir + 'loss.pickle'
        reward_file = sub_dir + 'reward.pickle'
        reward_clamped_file = sub_dir + 'reward_clamped.pickle'
        log_avg_episodes = 50

        best_score = 0
        best_score_clamped = 0
        avg_score = 0
        avg_score_clamped = 0
        loss_history = []
        reward_history = []
        reward_clamped_history = []

        # Initialize logfile with header
        with open(logfile, 'w') as fp:
            fp.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n' +
                     'Trained game:                       ' + str(self.game) + '\n' +
                     'Learning rate:                      ' + str(self.learning_rate) + '\n' +
                     'Batch size:                         ' + str(self.batch_size) + '\n' +
                     'Memory size(replay):                ' + str(self.mem_size) + '\n' +
                     'Pretrained:                         ' + str(self.pretrained_model) + '\n' +
                     'Started training after k frames:    ' + str(self.start_train_after) + '\n' +
                     'Optimized after k frames:           ' + str(self.optimize_each_k) + '\n' +
                     'Target net update after k frame:    ' + str(self.update_target_net_each_k_steps) + '\n\n' +
                     '------------------------------------------------------' +
                     '--------------------------------------------------\n')

        print('Started training...\nLogging to', sub_dir)

        for i_episode in range(1,num_episodes):
            # reset game at the start of each episode
            screen = self.env.reset()

            # list of k last frames
            last_k_frames = []
            for j in range(self.num_stored_frames):
                last_k_frames.append(None)
                last_k_frames[j] = gray2pytorch(screen)

            if i_episode == 1:
                self.replay.pushFrame(last_k_frames[0].cpu())

            # frame is saved as ByteTensor -> convert to gray value between 0 and 1
            state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0

            done = False # games end indicator variable
            # reset score with initial lives, because every lost live adds -1
            total_reward = self.env.get_lives()
            total_reward_clamped = self.env.get_lives()

            # Loop over one game
            while not done:
                self.steps +=1

                action = self.select_action(state)

                # perform selected action on game
                screen, reward, done, info = self.env.step(action[0,0])#envTest.step(action[0,0])
                total_reward += int(reward)

                #   clamp rewards
                reward = torch.Tensor([np.clip(reward,-1,1)])
                total_reward_clamped += int(reward[0])

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
                    loss = self.optimize(net_updates)

                    # Logging
                    loss_history.append(loss)
                    #q_history.append(q_value)
                    #exp_q_history.append(exp_q_value)

                    net_updates += 1

                # set current state to next state to select next action
                if next_state is not None:
                    state = next_state

                if self.use_cuda:
                    state = state.cuda()

                # plays episode until there are no more lives left ( == done)
                if done:
                    break;

            # Save rewards
            reward_history.append(total_reward)
            reward_clamped_history.append(total_reward_clamped)

            print('Episode: {:6} |  '.format(i_episode),
                  'steps {:8} |  '.format(self.steps),
                  'loss: {:.2E} |  '.format(loss if loss else 0),
                  'score: ({:4}/{:4}) |  '.format(total_reward_clamped,total_reward),
                  'best score: ({:4}/{:4}) |  '.format(best_score_clamped,best_score),
                  'replay size: {:7}'.format(len(self.replay)))

            avg_score_clamped += total_reward_clamped
            avg_score += total_reward
            if total_reward_clamped > best_score_clamped:
                best_score_clamped = total_reward_clamped
            if total_reward > best_score:
                best_score = total_reward

            if i_episode % log_avg_episodes == 0 and i_episode!=0:
                avg_score_clamped /= log_avg_episodes
                avg_score /= log_avg_episodes

                print('----------------------------------------------------------------'
                      '-----------------------------------------------------------------',
                      '\nLogging to file: \nEpisode: {:6}   '.format(i_episode),
                      'steps: {:8}   '.format(self.steps),
                      'avg on last {:4} games ({:6.1f}/{:6.1f})   '.format(log_avg_episodes, avg_score_clamped,avg_score),
                      'best score: ({:4}/{:4})'.format(best_score_clamped, best_score),
                      '\n---------------------------------------------------------------'
                      '------------------------------------------------------------------')
                # Logfile
                with open(logfile, 'a') as fp:
                    fp.write('Episode: {:6} |  '.format(i_episode) +
                             'steps: {:8} |  '.format(self.steps) +
                             'avg on last {:4} games ({:6.1f}/{:6.1f}) |  '.format(log_avg_episodes, avg_score_clamped,avg_score) +
                             'best score: ({:4}/{:4})\n'.format(best_score_clamped, best_score))
                # Dump loss & reward
                with open(loss_file, 'wb') as fp:
                    pickle.dump(loss_history, fp)
                with open(reward_file, 'wb') as fp:
                    pickle.dump(reward_history, fp)
                with open(reward_clamped_file, 'wb') as fp:
                    pickle.dump(reward_clamped_history, fp)

                avg_score_clamped = 0
                avg_score = 0

            if i_episode % self.save_net_each_k_episodes == 0:
                with open(logfile, 'a') as fp:
                    fp.write('Saved model at episode ' + str(i_episode) + '...\n')
                self.target_net.save(sub_dir + self.game + '-' + str(i_episode) + '_episodes.model')

        print('Training done!')
        self.target_net.save(sub_dir + self.game + '.model')
