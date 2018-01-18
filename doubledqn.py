#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class DOUBLEDQN(nn.Module):
    def __init__(self, channels_in, num_actions):
        super(DOUBLEDQN, self).__init__()

        # Subnet 1
        self.net1_conv1 = nn.Conv2d(in_channels=channels_in,
                          out_channels=32,
                          kernel_size=8,
                          stride=4)
        self.net1_conv2 = nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=4,
                          stride=2)
        self.net1_conv3 = nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=3,
                          stride=1)
        self.net1_fc4 = nn.Linear(in_features=64*7*7,
                          out_features=512)

        # Subnet2
        self.net2_conv1 = nn.Conv2d(in_channels=channels_in,
                          out_channels=32,
                          kernel_size=8,
                          stride=4)
        self.net2_conv2 = nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=4,
                          stride=2)
        self.net2_conv3 = nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=3,
                          stride=1)
        self.net2_fc4 = nn.Linear(in_features=64*7*7,
                          out_features=512)

        # Union net
        self.fc5 = nn.Linear(in_features=1024,
                          out_features=512)
        self.fc6 = nn.Linear(in_features=512,
                          out_features=num_actions)

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

        N, C, H, W = in1.size()

        # Subnet 1
        in1 = F.relu(self.net1_conv1(in1))
        in1 = F.relu(self.net1_conv2(in1))
        in1 = F.relu(self.net1_conv3(in1))
        in1 = in1.view(N,-1) # change the view from 2d to 1d
        in1 = F.relu(self.net1_fc4(in1))

        # Subnet 1
        in2 = F.relu(self.net2_conv1(in2))
        in2 = F.relu(self.net2_conv2(in2))
        in2 = F.relu(self.net2_conv3(in2))
        in2 = in2.view(N,-1) # change the view from 2d to 1d
        in2 = F.relu(self.net2_fc4(in2))

        x = torch.cat((in1,in2),dim = 1)

        # Union net
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
