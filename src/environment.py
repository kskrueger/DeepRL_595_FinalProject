import cv2
import gym
import numpy as np


class Environment(object):
    def __init__(self, env_name, args, test=False):
        self.env = gym.make(env_name)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def seed(self, seed):
        """
        Seed the randomness of the environment
        """
        self.env.seed(seed)

    def reset(self):
        """
        Reset environment and return observation
        """
        observation = self.env.reset()

        return np.array(observation)

    def step(self, action):
        """
        Take a step in the environment and return new observation
        """
        if not self.env.action_space.contains(action):
            raise ValueError('Invalid action!')

        observation, reward, done, info = self.env.step(action)

        # cv2.imshow("img", np.array(observation)[:, :, 3])
        # cv2.waitKey(1)

        return np.array(observation), reward, done, info

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()
