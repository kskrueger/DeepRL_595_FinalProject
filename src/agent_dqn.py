#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import random
from itertools import count

import numpy as np
from collections import deque, namedtuple
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
# matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
from torch import nn

from agent import Agent
from dqn_model import DQN

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize all parameters for training here
        """

        super(Agent_DQN, self).__init__(env)

        self.network_name = "Project4_0"  # increment this number before training again
        self.episode_durations = []
        self.logs = []
        self.scores = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.BATCH_SIZE = 32
        self.GAMMA = 0.99  # discount factor
        self.EPS_START = 1.0  # probability of choosing random action, start high for exploration
        self.EPS_END = .025  # low probability for random action means mostly exploitation at end
        self.EPS_DECAY = 10000000  # rate of decay, in STEPS
        self.EPS_STEP = (self.EPS_START - self.EPS_END) / self.EPS_DECAY
        self.TARGET_UPDATE = 5000  # how often to update target network (copies q_net weights)
        self.REPLAY_MEMORY_SIZE = 10000
        self.LEARNING_RATE = 1.5e-4
        self.NUM_EPISODES = 50000000
        self.SAVE_FREQ = 500
        self.START_TRAIN = 5000
        self.NETWORK_TRAIN_INTERVAL = 10
        self.eps_threshold = self.EPS_START

        self.steps_done = 0

        self.losses = []
        self.memory = deque([], maxlen=self.REPLAY_MEMORY_SIZE)
        self.env = env

        self.state = (np.zeros((6, 1)), np.zeros((720, 960, 3)), np.zeros((720, 960, 4)))

        # Get number of actions from gym action space
        self.n_actions = env.action_space.n

        # screen_height, screen_width
        num_frames = 4
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE)

        if args.test_dqn:
            print('loading trained model')

            self.policy_net.load_state_dict(torch.load("weightsFile", map_location=self.device))
            self.target_net.load_state_dict(torch.load("weightsFile", map_location=self.device))

    def init_game_setting(self):
        """
        Testing calls this at the beginning of new episode. Can initialize for tests anything here.
        """

        pass

    def make_action(self, observation, test=True):
        """
        Return action from network (with noise from EPS)
        """

        action = None

        return action

    def push(self, *args):
        """
        Add new data to the memory buffer
        """

        self.memory.append(Transition(*args))

    def replay_buffer(self, batch_size):
        """
        Get a random batch of data from memory buffer.
        """

        return random.sample(self.memory, batch_size)

    def train(self):
        def optimize_model():
            if len(self.memory) < self.BATCH_SIZE:
                return
            transitions = self.replay_buffer(self.BATCH_SIZE)
            # Batch-array of Transitions to Transition of batch-arrays
            batch = Transition(*zip(*transitions))

            # Compute mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                          dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
            # These are the actions which would've been taken for each batch state according to policy_net
            state_action_batch = self.policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) values for all next states.
            # Expected values of actions for non_final_next_states are computed based on the "older" target_net;
            # selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            # TODO: Remove the max() when doing continuous outputs
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            # Compute expected Q values
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_batch, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            return loss.cpu().detach().numpy()

        def plot_durations():
            plt.figure(2)
            plt.clf()
            durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
            scores_t = torch.tensor(self.scores, dtype=torch.float)
            plt.title('Training...')
            plt.xlabel('Episode')
            plt.ylabel('Duration')
            plt.plot(durations_t.numpy())
            # Take 100 episode averages and plot them too
            if len(durations_t) >= 100:
                means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                plt.plot(means.numpy())
                plt.plot(scores_t.numpy())

            plt.pause(0.001)  # pause a bit so that plots are updated

        try:
            for i_episode in range(self.NUM_EPISODES):
                # Init environment and state
                self.state = self.env.reset()
                self.state = torch.tensor(np.swapaxes(np.array([self.state]), 3, 1), dtype=torch.float32,
                                          device=self.device)

                score = 0
                for t in count():
                    # Select and perform action
                    action = self.make_action(self.state, test=False)
                    next_state, reward, done, _ = self.env.step(action.item())
                    score += reward
                    self.steps_done += 1

                    if self.steps_done > self.REPLAY_MEMORY_SIZE:
                        self.eps_threshold = max(self.EPS_END, self.EPS_START - (self.EPS_STEP * self.steps_done))

                    next_state = torch.tensor(np.swapaxes(np.array([next_state]), 3, 1), dtype=torch.float32,
                                              device=self.device)

                    reward = torch.tensor([reward], device=self.device)

                    # Observe new state
                    if done:
                        next_state = None

                    # Store transition in memory
                    self.push(self.state, action, next_state, reward)

                    # Move to next state
                    self.state = next_state

                    # Perform one step of optimization on policy network
                    if self.steps_done > self.START_TRAIN and self.steps_done % self.NETWORK_TRAIN_INTERVAL == 0:
                        loss = optimize_model()
                        self.losses.append(loss)

                    if done:
                        self.episode_durations.append(t + 1)
                        self.scores.append(score)

                        log_data = (i_episode, self.steps_done,
                                    t + 1, round(
                            np.mean(self.episode_durations[-100:]) if len(self.episode_durations) > 100 else 0, 2),
                                    round(self.eps_threshold, 4),
                                    score, round(np.mean(self.scores[-100:]) if len(self.scores) > 100 else 0, 2),
                                    np.round(loss, 4) if len(self.losses) > 0 else "None",
                                    "Training" if self.steps_done > self.START_TRAIN else "Fill Buffer")

                        print("Episode: {}, Steps: {}, Duration: {}, Avg Duration: {}, Eps: {}, Score: {}, "
                              "Avg Score: {}, Loss: {}, Mode: {}".format(*log_data))
                        self.logs.append(log_data)
                        # plot_durations()
                        break

                # Update target network, copying all weights and biases in DQN
                if i_episode % self.TARGET_UPDATE == 0:
                    print("Update target network!")
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if i_episode % self.SAVE_FREQ == 0:
                    torch.save(self.policy_net.state_dict(), self.network_name)
                    torch.save(self.target_net.state_dict(), self.network_name)
                    np.save("{}_durations".format(self.network_name), np.array(self.episode_durations))
                    np.save("{}_scores".format(self.network_name), np.array(self.scores))
                    np.save("{}_logs".format(self.network_name), np.array(self.logs))
                    print("Saved networks")
        except Exception as e:
            print(e)
            print("Cancelled")

        np.save("{}_durations".format(self.network_name), np.array(self.logs))
        # np.save("{}_durations".format(self.network_name), np.array(self.episode_durations))
        # np.save("{}_durations".format(self.network_name), np.array(self.scores))
