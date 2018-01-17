#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from agent import Agent

def main(args):
    agent = Agent(args.game1, args.game2, pretrained_model=args.pretrained_model)

    if args.play:
        agent.play()
    else:
        agent.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python main.py', description = 'DQN reinforcement learning agent')
    parser.add_argument('game1', help='The first game that will be used [Breakout-v0, SpaceInvaders-v0, ' +
                                      'Assault-v0, Phoenix-v0, Skiing-v0, Enduro-v0, BeamRider-v0]', type=str)
    parser.add_argument('game2', help='The first game that will be used [Breakout-v0, SpaceInvaders-v0, ' +
                                      'Assault-v0, Phoenix-v0, Skiing-v0, Enduro-v0, BeamRider-v0]', type=str)

    parser.add_argument('-t','--train', action='store_true',  help='The agent will be trained (default behavior)')
    parser.add_argument('-p','--play', action='store_true', help='The agent will play a game')
    parser.add_argument('-m','--pretrained_model', help='The agent will use the specified net', type=str)
    args = parser.parse_args()

    main(args)
