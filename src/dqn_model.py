#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=4):
        """
        Network is constructed here.
        """
        super(DQN, self).__init__()

        outputs = num_actions
        # (84, 84, 4)
        h = 84
        w = 84
        d = in_channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv1 = nn.Conv2d(d, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        # Number linear input connections depends on output of conv2d layers and img size
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3, stride=1)
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3, stride=1)
        linear_input_size = conv_h * conv_w * 64  # last_num_channels is 64
        self.hidden = nn.Linear(linear_input_size, 512)

        self.head = nn.Linear(512, outputs)

        # Can run the custom weight initialization method here
        # self.apply(self.init_weights)

    def forward(self, x):
        """
        Step the network forward
        """

        if type(x) is np.ndarray:
            x = np.array([x])
            x = x.swapaxes(1, 3).astype(np.float32)
            x = torch.from_numpy(x)
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.hidden(x.view(x.size(0), -1)))
        x = self.head(x)

        return x

    # Can add a custom weight initialization method here
    def init_weights(self, m):
        print("Initialized weights to custom method")
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.uniform(m.weight, -.01, .01)
            m.bias.data.fill_(.01)
