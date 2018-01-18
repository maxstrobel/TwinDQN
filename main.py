#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from argparse import RawTextHelpFormatter
from agent import Agent

def main(args):
    if args.games==1:
        game1='Breakout-v0'
        game2='SpaceInvaders-v0'
    if args.games==2:
        game1='SpaceInvaders-v0'
        game2='Assault-v0'
    if args.games==3:
        game1='SpaceInvaders-v0'
        game2='Phoenix-v0'
    if args.games==4:
        game1='Assault-v0'
        game2='Phoenix-v0'

    agent = Agent(game1, game2, pretrained_model=args.pretrained_model)

    if args.play:
        agent.play()
    else:
        agent.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python main.py', description = 'DQN reinforcement learning agent',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('games', type=int, choices=range(1,5), help='Select the both games, that should be played:\n' +
                                                                    '(1) Breakout and SpaceInvaders\n' +
                                                                    '(2) SpaceInvaders and Assault\n' +
                                                                    '(3) Breakout and Phoenix\n' +
                                                                    '(4) Assault and Phoenix')

    parser.add_argument('-t','--train', action='store_true',  help='The agent will be trained (default behavior)')
    parser.add_argument('-p','--play', action='store_true', help='The agent will play a game')
    parser.add_argument('-m','--pretrained_model', help='The agent will use the specified net', type=str)
    args = parser.parse_args()

    main(args)
