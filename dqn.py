"""DQN"""
"""
 Sources

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, channels_in, num_actions, input_h, input_w):
        super(DQN, self).__init__()
        
        # TODO: Make input more generic
        # Padding for first layer => input 84x84
        pad_h = 1
        pad_w = 6
    
        
        self.conv1 = nn.Conv2d(in_channels=channels_in,
                          out_channels=32,
                          kernel_size=8,
                          stride=4,
                          padding=(pad_h, pad_w))
        self.conv2 = nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=4,
                          stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=3,
                          stride=1)
        self.fc4 = nn.Linear(in_features=64*7*7,
                          out_features=512)
        self.fc5 = nn.Linear(in_features=512,
                          out_features=num_actions)


    def forward(self, x):
        """
        Forward pass of the dqn. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        N, C, H, W = x.size()
        #print('forward',H,W)
        x = F.relu(self.conv1(x))
        #print('conv1',x.size())
        x = F.relu(self.conv2(x))
        #print('conv2',x.size())
        x = F.relu(self.conv3(x))
        #print('conv3',x.size())
        x = x.view(N,-1) # change the view from 2d to 1d
        #print('conv3_flat', x.size())
        x = F.relu(self.fc4(x))
        #print('fc1',x.size())
        x = self.fc5(x)
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
