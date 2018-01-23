#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from argparse import RawTextHelpFormatter
from double_agent import DoubleAgent

def main(args):
    if args.games==1:
        game1='Breakout'
        game2='SpaceInvaders'
    if args.games==2:
        game1='SpaceInvaders'
        game2='Assault'
    if args.games==3:
        game1='SpaceInvaders'
        game2='Phoenix'
    if args.games==4:
        game1='Assault'
        game2='Phoenix'

    agent = DoubleAgent(game1, game2,
                        pretrained_model=args.m,
                        pretrained_subnet1=args.s1,
                        pretrained_subnet2=args.s2)

    if args.play:
        agent.play()
    else:
        agent.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python double_game.py', description = 'DQN reinforcement learning agent for playing two games simultaneously',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('games', type=int, choices=range(1,5), help='Select the both games, that should be played:\n' +
                                                                    '(1) Breakout and SpaceInvaders\n' +
                                                                    '(2) SpaceInvaders and Assault\n' +
                                                                    '(3) Breakout and Phoenix\n' +
                                                                    '(4) Assault and Phoenix')

    parser.add_argument('-t','--train', action='store_true',  help='The agent will be trained (default behavior)')
    parser.add_argument('-p','--play', action='store_true', help='The agent will play a game')
    parser.add_argument('-m', help='The agent will use the specified model for the whole net', type=str)
    parser.add_argument('-s1', help='The agent will use the specified net as subnet for game 1', type=str)
    parser.add_argument('-s2', help='The agent will use the specified net as subnet for game 2', type=str)
    args = parser.parse_args()

    main(args)
