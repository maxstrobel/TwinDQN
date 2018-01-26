#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, channels_in, num_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels_in,
                          out_channels=32,
                          kernel_size=8,
                          stride=4)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=4,
                          stride=2)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=3,
                          stride=1)
        self.relu3 = nn.ReLU(True)
        self.flat = Flatten()
        self.fc4 = nn.Linear(in_features=64*7*7,
                          out_features=512)
        self.relu4 = nn.ReLU(True)
        self.fc5 = nn.Linear(in_features=512,
                          out_features=num_actions)


    def forward(self, x):
        """
        Forward pass of the dqn. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flat(x) # change the view from 2d to 1d
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)

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
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
