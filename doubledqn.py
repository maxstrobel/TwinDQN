#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from dqn import DQN


class DOUBLEDQN(nn.Module):
    def __init__(self, channels_in, num_actions_first, num_actions_second):
        super(DOUBLEDQN, self).__init__()

        self.net1 = DQN(channels_in = channels_in,
                       num_actions = num_actions_first)

        self.net2 = DQN(channels_in = channels_in,
                       num_actions = num_actions_second)

        #for param in self.net1.parameters():
        #    param.requires_grad = False
        #for param in self.net2.parameters():
        #    param.requires_grad = False     

        self.fc5 = nn.Linear(in_features=1024,
                          out_features=512)
        self.fc6 = nn.Linear(in_features=512,
                          out_features=num_actions_second)

    def forward(self, x):
        """
        Forward pass of the dqn. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        inp = torch.chunk(x,8,dim=1)
        in2 = inp[1]
        in1 = inp[0]
        for i in range(1,4):
            in2 = torch.cat((in2,inp[2*i+1]),dim = 1)
            in1 = torch.cat((in1,inp[2*i]),dim = 1)
        N, C, H, W = x.size()

        in1 = F.relu(self.net1.conv1(in1))
        in2 = F.relu(self.net2.conv1(in2))

        in1 = F.relu(self.net1.conv2(in1))
        in2 = F.relu(self.net2.conv2(in2))

        in1 = F.relu(self.net1.conv3(in1))
        in2 = F.relu(self.net2.conv3(in2))

        in1 = in1.view(N,-1)
        in2 = in2.view(N,-1)

        in1 = F.relu(self.net1.fc4(in1))
        in2 = F.relu(self.net2.fc4(in2))

        x = torch.cat((in1,in2),dim = 1)

        #x = x.view(N,-1) # change the view from 2d to 1d

        x = F.relu(self.fc5(x))
        x = self.fc6(x)


        return x


    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda


    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self.state_dict(), path)


    def load(self, path):
        """
        Load model with its parameters from the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Loading model... %s' % path)
        self.load_state_dict(torch.load(path))
