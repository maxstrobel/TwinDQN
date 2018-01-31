#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
from random import random, randrange
from collections import namedtuple
from datetime import datetime
import pickle
import os

from environment import Environment
from twin_dqn import TwinDQN
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
dimensions = {'Breakout': (32, 195, 8, 152),
              'SpaceInvaders': (21, 195, 20, 141),
              'Assault': (50, 240, 5, 155),
              'Phoenix': (23, 183, 0, 160),
              'Skiing': (55, 202, 8, 152),
              'Enduro': (50, 154, 8, 160),
              'BeamRider': (32, 180, 9, 159),
              }

game_name = {'Breakout': 'BreakoutNoFrameskip-v4',
             'SpaceInvaders': 'SpaceInvadersNoFrameskip-v4',
             'Assault': 'AssaultNoFrameskip-v4',
             'Phoenix': 'PhoenixNoFrameskip-v4',
             'Skiing': 'SkiingNoFrameskip-v4',
             'Enduro': 'EnduroNoFrameskip-v4',
             'BeamRider': 'BeamRiderNoFrameskip-v4',
              }


class DoubleAgent(object):
    def __init__(self,
                 game1,
                 game2,
                 mem_size = 1000000,
                 state_buffer_size = 4,
                 batch_size = 64,
                 learning_rate = 1e-5,
                 pretrained_model = None,
                 pretrained_subnet1 = False,
                 pretrained_subnet2 = False,
                 frameskip = 4,
                 frozen = False
                 ):
        """
        Inputs:
        - game 1: string to select the game 1
        - game 2: string to select the game 2
        - mem_size: int length of the replay memory
        - state_buffer_size: int number of recent frames used as input for neural network
        - batch_size: int
        - learning_rate: float
        - pretrained_model: str path to the model
        - pretrained_subnet1: str path to the model of the subnet
        - pretrained_subnet2: str path to the model of the subnet
        - frozen: boolean freeze pretrained subnets
        """

        # Namestring
        self.game1 = game1
        self.game2 = game2

        # Environment
        self.env1 = Environment(game_name[game1], dimensions[game1], frameskip=frameskip)
        self.env2 = Environment(game_name[game2], dimensions[game2], frameskip=frameskip)


        # Neural net
        self.pretrained_subnet1 = pretrained_subnet1
        self.pretrained_subnet2 = pretrained_subnet2
        self.net = TwinDQN(channels_in = state_buffer_size,
                             num_actions = self.env2.get_number_of_actions(),
                             pretrained_subnet1 = pretrained_subnet1,
                             pretrained_subnet2 = pretrained_subnet2,
                             frozen = frozen)
        self.target_net = TwinDQN(channels_in = state_buffer_size,
                                    num_actions = self.env2.get_number_of_actions(),
                                    pretrained_subnet1 = pretrained_subnet1,
                                    pretrained_subnet2 = pretrained_subnet2,
                                    frozen = frozen)

        # Cuda
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.net.cuda()
            self.target_net.cuda()

        # Pretrained
        if pretrained_model:
            self.net.load(pretrained_model)
            self.target_net.load(pretrained_model)
            self.pretrained_model = True
        else:
            self.pretrained_model = False

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                    lr=learning_rate)
        #self.optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.net.parameters()),
        #                               lr=learning_rate,alpha=0.95, eps=0.01)

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
        - action: int
        """
        # Hyperparameters
        EPSILON_START = 1
        EPSILON_END = 0.1
        EPSILON_DECAY = 1000000
        EPSILON_PLAY = 0.01
        MAXNOOPS = 30

        # Decrease of epsilon value
        if not self.pretrained_model:
            #epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            #                        np.exp(-1. * (self.steps-self.batch_size) / EPSILON_DECAY)
            epsilon = EPSILON_START - self.steps * (EPSILON_START - EPSILON_END) / EPSILON_DECAY
        elif mode=='play':
            epsilon = EPSILON_PLAY
        else:
            epsilon = EPSILON_END

        if epsilon < random():
            # Action according to neural net
            # Wrap tensor into variable
            state_variable = Variable(observation, volatile=True)

            # Evaluate network and return action with maximum of activation
            action = self.net(state_variable).data.max(1)[1].view(1,1)

            # Prevent noops
            if action[0,0]!=1:
                self.noops_count += 1
                if self.noops_count == MAXNOOPS:
                    action[0,0] = 1
                    self.noops_count = 0
            else:
                self.noops_count = 0
        else:
            # Random action
            action = self.env2.sample_action()
            action = LongTensor([[action]])

        return action


    def map_action(self, action):
        """
        Maps action from game with more actions
        to game with less actions

        Inputs:
        - action: int
        Returns:
        - action: int
        """
        # Map SpaceInvaders on Breakout
        if self.game1=='Breakout' and self.game2=='SpaceInvaders':
            if action>3: # shoot+right/left --> right/left
                return action-2

        # Map Assault on SpaceInvaders
        if self.game1=='SpaceInvaders' and self.game2=='Assault':
            if action!=0: # all actions except 2nd idle
                return action-1

        # Map Phoenix on SpaceInvaders
        if self.game1=='SpaceInvaders' and self.game2=='Phoenix':
            if action==4: # shield --> idle
                return 0
            if action==7: # shield+shot --> shot
                return 1
            if action>4: # shoot+right/left --> shoot+right/left
                return action-1

        # Map Phoenix on Assault
        if self.game1=='Assault' and self.game2=='Phoenix':
            if action==4: # shield --> idle
                return 0
            if action==7: # shield+shot --> shot
                return 2
            if 1<= action and action<=3: # shot/right/left --> shot/right/left
                return action+1

        # No mapping necessary
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

        # Score counter
        total_reward_game1 = 0
        total_reward_game2 = 0
        total_reward = 0

        # Reset game
        screen1 = self.env1.reset()
        screen2 = self.env2.reset()

        # list of k last frames
        last_k_frames = []
        for j in range(self.num_stored_frames):
            last_k_frames.append(None)
            last_k_frames[j] = torch.cat((gray2pytorch(screen1),gray2pytorch(screen2)),dim=1)

        # frame is saved as ByteTensor -> convert to gray value between 0 and 1
        state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0

        while not done:
            action = self.select_action(state, mode='play')[0,0]
            action1 = self.map_action(action)
            action2 = action

            # perform selected action on game
            screen1, reward1, _, done1, _ = self.env1.step(action1, mode='play')
            screen2, reward2, _, done2, _ = self.env2.step(action2, mode='play')

            # Logging
            total_reward_game1 += int(reward1)
            total_reward_game2 += int(reward2)
            total_reward += int(reward1) + int(reward2)

            # save latest frame, discard oldest
            for j in range(self.num_stored_frames-1):
                last_k_frames[j] = last_k_frames[j+1]
            last_k_frames[self.num_stored_frames-1] = torch.cat((gray2pytorch(screen1),gray2pytorch(screen2)),dim=1)

            # convert frames to range 0 to 1 again
            state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0

            # Merged game over indicator
            done = done1 or done2
        print('Final score {}: {}'.format(self.game1, total_reward_game1))
        print('Final score {}: {}'.format(self.game2, total_reward_game2))
        print('Final score (total): {}'.format(total_reward))
        self.env1.game.close()
        self.env2.game.close()


    def play_stats(self, n_games, mode='random'):
        """
        Play N games randomly or evaluate a net and log results for statistics

        Input:
        - n_games: int Number of games to play
        - mode: str 'random' or 'evaluation'
        """
        # Subdirectory for logging
        sub_dir = mode + '_' + self.game1 + '+' + self.game2 + '/'
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        # Store history total
        reward_history = []
        reward_clamped_history = []
        # Store history game 1
        reward_history_game1 = []
        reward_clamped_history_game1 = []
        # Store history game 2
        reward_history_game2 = []
        reward_clamped_history_game2 = []

        # Number of actions to sample from
        n_actions = self.env2.get_number_of_actions()

        for i_episode in range(1, n_games+1):
            # Reset game
            screen1 = self.env1.reset()
            screen2 = self.env2.reset()

            # Store screen
            if mode=='evaluation':
                # list of k last frames
                last_k_frames = []
                for j in range(self.num_stored_frames):
                    last_k_frames.append(None)
                    last_k_frames[j] = torch.cat((gray2pytorch(screen1),gray2pytorch(screen2)),dim=1)
                # frame is saved as ByteTensor -> convert to gray value between 0 and 1
                state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0

            # games end indicator variable
            done = False

            # reset score with initial lives, because every lost live adds -1
            total_reward_game1 = 0
            total_reward_clamped_game1 = self.env1.get_lives()
            total_reward_game2 = 0
            total_reward_clamped_game2 = self.env2.get_lives()
            # total scores for both games
            total_reward = total_reward_game1 + total_reward_game2
            total_reward_clamped = total_reward_clamped_game1 + total_reward_clamped_game2

            while not done:
                if mode=='random':
                    action = randrange(n_actions)
                elif mode=='evaluation':
                    action = self.select_action(state, mode='play')[0,0]
                action1 = self.map_action(action)
                action2 = action

                screen1, reward1, reward1_clamped, done1, _ = self.env1.step(action1)
                screen2, reward2, reward2_clamped, done2, _ = self.env2.step(action2)

                # Logging
                total_reward_game1 += int(reward1)
                total_reward_game2 += int(reward2)
                total_reward += int(reward1) + int(reward2)
                total_reward_clamped_game1 += reward1_clamped
                total_reward_clamped_game2 += reward2_clamped
                total_reward_clamped += reward1_clamped + reward2_clamped

                if mode=='evaluation':
                    # save latest frame, discard oldest
                    for j in range(self.num_stored_frames-1):
                        last_k_frames[j] = last_k_frames[j+1]
                    last_k_frames[self.num_stored_frames-1] = torch.cat((gray2pytorch(screen1),gray2pytorch(screen2)),dim=1)

                    # convert frames to range 0 to 1 again
                    state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0

                # Merged game over indicator
                done = done1 or done2

            # Print current result
            print('Episode: {:6}/{:6} |   '.format(i_episode, n_games) +
                  'score total: ({:6.1f}/{:7.1f}) |   '.format(total_reward_clamped,total_reward) +
                  'score game1: ({:6.1f}/{:7.1f}) |   '.format(total_reward_clamped_game1,total_reward_game1) +
                  'score game2: ({:6.1f}/{:7.1f})'.format(total_reward_clamped_game2,total_reward_game2))

            # Save rewards
            reward_history_game1.append(total_reward_game1)
            reward_history_game2.append(total_reward_game2)
            reward_history.append(total_reward)
            reward_clamped_history_game1.append(total_reward_clamped_game1)
            reward_clamped_history_game2.append(total_reward_clamped_game2)
            reward_clamped_history.append(total_reward_clamped)

        avg_reward_total = np.sum(reward_history) / len(reward_history)
        avg_reward_total_clamped = np.sum(reward_clamped_history) / len(reward_clamped_history)
        avg_reward_game1 = np.sum(reward_history_game1) / len(reward_history_game1)
        avg_reward_game1_clamped = np.sum(reward_clamped_history_game1) / len(reward_clamped_history_game1)
        avg_reward_game2 = np.sum(reward_history_game2) / len(reward_history_game2)
        avg_reward_game2_clamped = np.sum(reward_clamped_history_game2) / len(reward_clamped_history_game2)

        # Print final result
        print('\n\n===========================================\n' +
              'avg score after {:6} episodes:\n'.format(n_games) +
              'avg total: ({:6.1f}/{:7.1f})\n'.format(avg_reward_total_clamped,avg_reward_total) +
              'avg game1: ({:6.1f}/{:7.1f})\n'.format(avg_reward_game1_clamped,avg_reward_game1) +
              'avg game2: ({:6.1f}/{:7.1f})\n'.format(avg_reward_game2_clamped,avg_reward_game2))

        # Log results to files
        with open(sub_dir + mode + '.txt', 'w') as fp:
            fp.write('avg score after {:6} episodes:\n'.format(n_games) +
                     'avg total: ({:6.1f}/{:7.1f})\n'.format(avg_reward_total_clamped,avg_reward_total) +
                     'avg game1: ({:6.1f}/{:7.1f})\n'.format(avg_reward_game1_clamped,avg_reward_game1) +
                     'avg game2: ({:6.1f}/{:7.1f})\n'.format(avg_reward_game2_clamped,avg_reward_game2))

        # Dump reward
        with open(sub_dir + mode + '_reward_game1.pickle', 'wb') as fp:
            pickle.dump(reward_history_game1, fp)
        with open(sub_dir + mode + '_reward_game2.pickle', 'wb') as fp:
            pickle.dump(reward_history_game2, fp)
        with open(sub_dir + mode + '_reward_total.pickle', 'wb') as fp:
            pickle.dump(reward_history, fp)

        with open(sub_dir + mode + '_reward_clamped_game1', 'wb') as fp:
            pickle.dump(reward_clamped_history_game1, fp)
        with open(sub_dir + mode + '_reward_clamped_game2', 'wb') as fp:
            pickle.dump(reward_clamped_history_game2, fp)
        with open(sub_dir + mode + '_reward_clamped_total', 'wb') as fp:
            pickle.dump(reward_clamped_history, fp)


    def train(self):
        """
        Train the agent
        """
        num_episodes = 100000
        net_updates = 0

        # Logging
        sub_dir = self.game1 + '+' + self.game2 + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '/'
        os.makedirs(sub_dir)
        logfile = sub_dir + 'train.txt'
        reward_file = sub_dir + 'reward.pickle'
        reward_file_game1 = sub_dir + 'reward_game1.pickle'
        reward_file_game2 = sub_dir + 'reward_game2.pickle'
        reward_clamped_file = sub_dir + 'reward_clamped.pickle'
        reward_clamped_file_game1 = sub_dir + 'reward_clamped_game1.pickle'
        reward_clamped_file_game2 = sub_dir + 'reward_clamped_game2.pickle'
        reward_clamped_file = sub_dir + 'reward_clamped.pickle'
        log_avg_episodes = 50

        # Total scores
        best_score = 0
        best_score_clamped = 0
        avg_score = 0
        avg_score_clamped = 0
        reward_history = []
        reward_clamped_history = []
        # Scores game 1
        avg_score_game1 = 0
        avg_score_clamped_game1 = 0
        reward_history_game1 = []
        reward_clamped_history_game1 = []
        # Scores game 2
        avg_score_game2 = 0
        avg_score_clamped_game2 = 0
        reward_history_game2 = []
        reward_clamped_history_game2 = []

        # Initialize logfile with header
        with open(logfile, 'w') as fp:
            fp.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n' +
                     'Trained game (first):               {}\n'.format(self.game1) +
                     'Trained game (second):              {}\n'.format(self.game2) +
                     'Learning rate:                      {:.2E}\n'.format(self.learning_rate) +
                     'Batch size:                         {:d}\n'.format(self.batch_size) +
                     'Memory size(replay):                {:d}\n'.format(self.mem_size) +
                     'Pretrained:                         {}\n'.format(self.pretrained_model) +
                     'Pretrained subnet 1:                {}\n'.format(self.pretrained_subnet1) +
                     'Pretrained subnet 2:                {}\n'.format(self.pretrained_subnet2) +
                     'Started training after k frames:    {:d}\n'.format(self.start_train_after) +
                     'Optimized after k frames:           {:d}\n'.format(self.optimize_each_k) +
                     'Target net update after k frame:    {:d}\n\n'.format(self.update_target_net_each_k_steps) +
                     '--------+-----------+----------------------+------------' +
                     '----------+----------------------+--------------------\n' +
                     'Episode | Steps     | ' +
                     '{:3} games avg total  | '.format(log_avg_episodes) +
                     '{:3} games avg game1  | '.format(log_avg_episodes) +
                     '{:3} games avg game2  | '.format(log_avg_episodes) +
                     'best score total \n' +
                     '--------+-----------+----------------------+------------' +
                     '----------+----------------------+--------------------\n')

        print('Started training...\nLogging to {}\n'.format(sub_dir) +
              'Episode | Steps     |   score total        |   score game 1       |   ' +
              'score game 2       | best score total')

        for i_episode in range(1,num_episodes):
            # reset game at the start of each episode
            screen1 = self.env1.reset()
            screen2 = self.env2.reset()

            # list of k last frames
            last_k_frames = []
            for j in range(self.num_stored_frames):
                last_k_frames.append(None)
                last_k_frames[j] = torch.cat((gray2pytorch(screen1),
                                              gray2pytorch(screen2)), dim=1)

            if i_episode == 1:
                self.replay.pushFrame(last_k_frames[0].cpu())

            # frame is saved as ByteTensor -> convert to gray value between 0 and 1
            state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0

            # games end indicator variable
            done1 = False
            done2 = False

            # reset score with initial lives, because every lost live adds -1
            total_reward_game1 = 0
            total_reward_clamped_game1 = self.env1.get_lives()
            total_reward_game2 = 0
            total_reward_clamped_game2 = self.env2.get_lives()
            # total scores for both games
            total_reward = total_reward_game1 + total_reward_game2
            total_reward_clamped = total_reward_clamped_game1 + total_reward_clamped_game2

            # Loop over one game
            while not done1 and not done2:
                self.steps +=1

                action = self.select_action(state)
                action1 = self.map_action(action[0,0])
                action2 = action[0,0]

                # perform selected action on game
                screen1, reward1, reward1_clamped, done1, _ = self.env1.step(action1)
                screen2, reward2, reward2_clamped, done2, _ = self.env2.step(action2)

                # Logging
                total_reward_game1 += int(reward1)
                total_reward_game2 += int(reward2)
                total_reward += int(reward1) + int(reward2)
                total_reward_clamped_game1 += reward1_clamped
                total_reward_clamped_game2 += reward2_clamped
                total_reward_clamped += reward1_clamped + reward2_clamped

                # Bake reward into tensor
                reward = torch.FloatTensor([reward1_clamped+reward2_clamped])

                #   save latest frame, discard oldest
                for j in range(self.num_stored_frames-1):
                    last_k_frames[j] = last_k_frames[j+1]
                last_k_frames[self.num_stored_frames-1] = torch.cat((gray2pytorch(screen1),
                                                                     gray2pytorch(screen2)), dim=1)

                # convert frames to range 0 to 1 again
                if not done1 and not done2:
                    next_state = torch.cat(last_k_frames,1).type(FloatTensor)/255.0
                else:
                    next_state = None

                # Store transition
                self.replay.pushFrame(last_k_frames[self.num_stored_frames - 1].cpu())
                self.replay.pushTransition((self.replay.getCurrentIndex()-1)%self.replay.capacity,
                                            action, reward, done1 or done2)

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
                if done1 or done2:
                    break;

            # Save rewards
            reward_history_game1.append(total_reward_game1)
            reward_history_game2.append(total_reward_game2)
            reward_history.append(total_reward)
            reward_clamped_history_game1.append(total_reward_clamped_game1)
            reward_clamped_history_game2.append(total_reward_clamped_game2)
            reward_clamped_history.append(total_reward_clamped)

            # Sum up for averages
            avg_score_clamped_game1 += total_reward_clamped_game1
            avg_score_clamped_game2 += total_reward_clamped_game2
            avg_score_clamped += total_reward_clamped
            avg_score_game1 += total_reward_game1
            avg_score_game2 += total_reward_game2
            avg_score += total_reward

            if total_reward_clamped > best_score_clamped:
                best_score_clamped = total_reward_clamped
            if total_reward > best_score:
                best_score = total_reward

            print('{:7} | '.format(i_episode) +
                  '{:9} |     '.format(self.steps) +
                  '({:6.1f}/{:7.1f}) |     '.format(total_reward_clamped,total_reward) +
                  '({:6.1f}/{:7.1f}) |     '.format(total_reward_clamped_game1,total_reward_game1) +
                  '({:6.1f}/{:7.1f}) |  '.format(total_reward_clamped_game2,total_reward_game2) +
                  '({:6.1f}/{:8.1f})'.format(best_score_clamped, best_score))

            if i_episode % log_avg_episodes == 0 and i_episode!=0:
                avg_score_clamped_game1 /= log_avg_episodes
                avg_score_clamped_game2 /= log_avg_episodes
                avg_score_clamped /= log_avg_episodes
                avg_score_game1 /= log_avg_episodes
                avg_score_game2 /= log_avg_episodes
                avg_score /= log_avg_episodes

                print('--------+-----------+----------------------+------------' +
                     '----------+----------------------+--------------------\n' +
                      'Episode | Steps     | ' +
                      '{:3} games avg total  | '.format(log_avg_episodes) +
                      '{:3} games avg game1  | '.format(log_avg_episodes) +
                      '{:3} games avg game2  | '.format(log_avg_episodes) +
                      'best score total \n' +
                      '{:7} | '.format(i_episode) +
                      '{:9} |     '.format(self.steps) +
                      '({:6.1f}/{:7.1f}) |     '.format(avg_score_clamped,avg_score) +
                      '({:6.1f}/{:7.1f}) |     '.format(avg_score_clamped_game1,avg_score_game1) +
                      '({:6.1f}/{:7.1f}) |  '.format(avg_score_clamped_game2,avg_score_game2) +
                      '({:6.1f}/{:8.1f})\n'.format(best_score_clamped, best_score) +
                      '\nLogging to file...\n\n'
                      '--------+-----------+----------------------+------------' +
                      '----------+----------------------+--------------------\n' +
                      'Episode | Steps     |   score total        |   score game 1       |   ' +
                      'score game 2       | best score total')
                # Logfile
                with open(logfile, 'a') as fp:
                    fp.write('{:7} | '.format(i_episode) +
                             '{:9} |     '.format(self.steps) +
                             '({:6.1f}/{:7.1f}) |     '.format(avg_score_clamped,avg_score) +
                             '({:6.1f}/{:7.1f}) |     '.format(avg_score_clamped_game1,avg_score_game1) +
                             '({:6.1f}/{:7.1f}) |  '.format(avg_score_clamped_game2,avg_score_game2) +
                             '({:6.1f}/{:8.1f})\n'.format(best_score_clamped, best_score))
                # Dump reward
                with open(reward_file_game1, 'wb') as fp:
                    pickle.dump(reward_history_game1, fp)
                with open(reward_file_game2, 'wb') as fp:
                    pickle.dump(reward_history_game2, fp)
                with open(reward_file, 'wb') as fp:
                    pickle.dump(reward_history, fp)

                with open(reward_clamped_file_game1, 'wb') as fp:
                    pickle.dump(reward_clamped_history_game1, fp)
                with open(reward_clamped_file_game2, 'wb') as fp:
                    pickle.dump(reward_clamped_history_game2, fp)
                with open(reward_clamped_file, 'wb') as fp:
                    pickle.dump(reward_clamped_history, fp)

                avg_score_clamped_game1 = 0
                avg_score_clamped_game2 = 0
                avg_score_clamped = 0
                avg_score_game1 = 0
                avg_score_game2 = 0
                avg_score = 0

            if i_episode % self.save_net_each_k_episodes == 0:
                with open(logfile, 'a') as fp:
                    fp.write('Saved model at episode ' + str(i_episode) + '...\n')
                self.target_net.save(sub_dir + str(i_episode) + '_episodes.model')

        print('Training done!')
        self.target_net.save(sub_dir + 'final.model')
