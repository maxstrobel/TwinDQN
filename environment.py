#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import numpy as np
import matplotlib.pyplot as plt
from cv2 import resize

class Environment(object):
    def __init__(self, game, inner_dimensions, width=84, height=84, record=False, seed=0):
        """
        Inputs:
        - game: string to select the game
        - inner_dimensions: tuple (h1,h2,w1,w2) with dimensions of the game (to crop borders)
                    breakout: (32, 195, 8, 152) -> exact dimensions
        - downsampling_rate: int 
        - record: boolean to enable record option
        - seed: int to reproduce results
        """
        # Setup game
        self.game = gym.make(game)
        self.game.seed(seed)
        self.game.reset()

        # Preprocessing parameter
        self.h1, self.h2, self.w1, self.w2 = inner_dimensions

        # Environment parameter
        self.width = width
        self.height = height


    def play_random(self, mode='human'):
        """
        Plays with random actions in the given environment

        Inputs:
        - mode: string to select game mode
                'human': window rendered with live game
                'rgb_array': preprocessed images rendered                
        """
        observation = self.game.reset()
        if mode == 'rgb_array':
            observation = self.preprocess(observation)
            plt.imshow(observation, cmap='gray')
            plt.show()

        while True:
            observation = self.game.render(mode=mode)
            if mode == 'rgb_array':
                observation = self.preprocess(observation)
                plt.imshow(observation, cmap='gray')
                plt.show()

            action = self.game.action_space.sample()
            observation, reward, done, info = self.game.step(action)
            if done:
                break
        self.game.close()


    def rgb2gray(self, img):
        """
        Transforms rgb image to gray value image via the mean of the channels
        Inputs:
        - image: np.array to process

        Returns:
        - image: np.array with gray value image
        """
        return np.mean(img, axis=2).astype(np.uint8)


    def crop(self, img):
        """
        Crops the image according to the games inner dimensions
        Inputs:
        - image: np.array to process

        Returns:
        - image: np.array with croped image
        """
        return img[self.h1:self.h2, self.w1:self.w2]


    def downsample(self, img):
        """
        Samples the image down according to the given downsampling rate
        Inputs:
        - image: np.array to process

        Returns:
        - image: np.array with downsampled image
        """
        return resize(img, (self.height, self.width))


    def preprocess(self, img):
        """
        Preprocessing pipeline
        Inputs:
        - image: np.array to process

        Returns:
        - image: np.array with preprocessed image
        """
        return self.rgb2gray(self.downsample(self.crop(img)))


    def reset(self):
        """
        Resets the environment

        Returns:
        - observation: np.array with initial observation (preprocessed)
        """
        observation = self.game.reset()
        observation = self.preprocess(observation)
        return observation


    def get_observation(self):
        """
        Returns the current observation

        Returns:
        - observation: np.array with current observation (preprocessed) 
        """
        observation = self.game.render(mode='rgb_array')
        observation = self.preprocess(observation)
        return observation


    def plot_observation(self):
        """
        Plots the current observation 
        """
        observation = self.game.render(mode='rgb_array')
        processed_observation = self.preprocess(observation)
        plt.figure()

        plt.subplot(1,2,1)
        plt.title('Observation')
        plt.imshow(observation)

        plt.subplot(1,2,2)
        plt.title('Preprocessed observation')
        plt.imshow(processed_observation, cmap='gray')

        plt.show()


    def step(self, action: int):
        """
        Executes a step in the environment

        Returns:

        - observation: np.array with current observation (preprocessed) 
        - reward: int
        - done: boolean to signal end of game
        - info: dict with the current number of lives
        """
        observation, reward, done, info = self.game.step(action)
        observation = self.preprocess(observation)
        return observation, reward, done, info


    def sample_action(self):
        """
        Sample a random action from action space

        Returns:
        actions: int
        """
        return self.game.action_space.sample()


    def get_actions(self):
        """
        Returns the executable actions of the current environment
        
        Returns:
        - actions: list of strings with the actions
        """
        return self.game.env.unwrapped.get_action_meanings()


    def get_number_of_actions(self):
        """
        Returns the number actions of the current environment

        Returns:
        - number_of_actions: int
        """
        return self.game.action_space.n


    def get_width(self):
        """
        Returns the width of the environment

        Returns:
        - width: int
        """
        return self.width


    def get_height(self):
        """
        Returns the height of the environment

        Returns:
        - height: int
        """
        return self.height


    def get_lives(self):
        """
        Returns the current number of lives

        Returns:
        - lives: int
        """
        return self.game.unwrapped.ale.lives()
