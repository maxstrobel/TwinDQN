#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from dqn import DQN

class DoubleDQN(nn.Module):
    def __init__(self,
                 channels_in,
                 num_actions,
                 pretrained_subnet1=False,
                 pretrained_subnet2=False,
                 frozen=False):
        super(DoubleDQN, self).__init__()

        # Subnet 1
        subnet1 = self.load_subnet(channels_in=channels_in, pretrained_net=pretrained_subnet1)
        feats_subnet1 = list(subnet1.children())
        self.subnet1 = nn.Sequential(*feats_subnet1[0:9])

        # Subnet 2
        subnet2 = self.load_subnet(channels_in=channels_in, pretrained_net=pretrained_subnet2)
        feats_subnet2 = list(subnet2.children())
        self.subnet2 = nn.Sequential(*feats_subnet2[0:9])

        # Freeze weights from pretrained models
        if frozen:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.requires_grad = False
            print('Subnets weights frozen')

        # Union net
        self.fc5 = nn.Linear(in_features=1024,
                             out_features=512)
        self.relu5 = nn.ReLU(True)
        self.fc6 = nn.Linear(in_features=512,
                             out_features=num_actions)


    def load_subnet(self, channels_in, pretrained_net=None):
        """
        Loads subnet
        If there is a pretrained model, its parameters are used

        Inputs:
        - channels_in: int

        Returns:
        - subnet
        """
        subnet = DQN(channels_in=channels_in, num_actions=1)
        if pretrained_net:
            pretrained_dict = torch.load(pretrained_net,
                                          map_location=lambda storage,
                                          loc: storage)
            subnet_dict = subnet.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in subnet_dict
                               and v.size() == subnet_dict[k].size()}
            # 2. overwrite entries in the existing state dict
            subnet_dict.update(pretrained_dict) 
            # 3. load the new state dict
            subnet.load_state_dict(subnet_dict)
            print('Loaded pretrained subnet...')

        return subnet


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

        # Subnet 1
        in1 = self.subnet1(in1)

        # Subnet 2
        in2 = self.subnet2(in2)

        x = torch.cat((in1,in2),dim = 1)

        # Union net
        x = self.fc5(x)
        x = self.relu5(x)
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
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
