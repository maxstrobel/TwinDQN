#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from argparse import RawTextHelpFormatter
from single_agent import SingleAgent


def main(args):
    if args.game==1:
        game='Breakout'
    if args.game==2:
        game='SpaceInvaders'
    if args.game==3:
        game='Phoenix'
    if args.game==4:
        game='Assault'

    agent = SingleAgent(game, pretrained_model=args.model)

    if args.play:
        agent.play(n=args.play)
    elif args.random:
        agent.play_stats(args.random, mode ='random')
    elif args.eval:
        agent.play_stats(args.eval, mode='evaluation')
    else:
        agent.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python single_game.py', description = 'DQN reinforcement learning agent for ATARI games',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('game', type=int, choices=range(1,5), help='Select the game, that should be played:\n' +
                                                                    '(1) Breakout\n' +
                                                                    '(2) SpaceInvaders\n' +
                                                                    '(3) Phoenix\n' +
                                                                    '(4) Assault')

    parser.add_argument('-t','--train', action='store_true',  help='The agent will be trained (default behavior)')
    parser.add_argument('-m','--model', help='The agent will use the specified net', type=str)
    parser.add_argument('-p','--play', metavar='N', type=int, help='The agent will play N games')
    parser.add_argument('-e','--eval', metavar='N', type=int, help='Play for N episodes with the loaded net and ' +
                                                                   'log avg results for statistics')
    parser.add_argument('-r','--random', metavar='N', type=int, help='Play for N episodes randomly and ' +
                                                                     'log avg results for statistics')
    args = parser.parse_args()

    main(args)
