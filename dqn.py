"""DQN"""
"""
 Sources

"""
import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, num_actions=4):
        super(DQN, self).__init__()

        # Input 84x84
        self.conv_block1 = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=32,
                          kernel_size=8,
                          stride=4),
                nn.ReLU(inplace=True)
                )
        self.conv_block2 = nn.Sequential(
                nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=4,
                          stride=2),
                nn.ReLU(inplace=True)
                )
        self.conv_block3 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=3,
                          stride=1),
                nn.ReLU(inplace=True)
                )
        self.fc1 = nn.Sequential(
                nn.Linear(in_features=64*7*7,
                          out_features=512),
                nn.ReLU(inplace=True)
                )
        self.fc2 = nn.Linear(
                in_features=512,
                out_features=num_actions
                )


    def forward(self, x):
        """
        Forward pass of the dqn. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        (N, C, H, W) = x.size()
        
        x = self.conv_block1(x)
        #print('conv1',x.size())
        x = self.conv_block2(x)
        #print('conv2',x.size())
        x = self.conv_block3(x)
        #print('conv3',x.size())
        x = x.view(N,-1) # change the view from 2d to 1d
        #print('conv3_flat', x.size())
        x = self.fc1(x)
        #print('fc1',x.size())
        x = self.fc2(x)
        #print('score',x.size())

        
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
        torch.save(self, path)
