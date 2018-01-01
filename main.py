#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 13:42:45 2017

@author: max
"""

from agent import Agent

def main():
    agent = Agent('Breakout-v0', (32, 195, 8, 152), downsampling_rate=1)
    agent.dqn_learning()

if __name__ == '__main__':
    main()