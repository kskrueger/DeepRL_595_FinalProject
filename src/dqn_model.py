#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self):
        """
        Network is constructed here.
        """
        super(DQN, self).__init__()

        # 1 continuous action for each joint (-1 to 1), plus the gripper (<.5 or >.5)
        self.NUM_ACTIONS = 8
        self.OVERHEAD_CAM_SHAPE = (720, 1280, 4)  # RGB-D Overhead Camera
        self.WRIST_CAM_SHAPE = (720, 1280, 3)  # RGB Wrist Camera
        self.MOTOR_SIGNAL_SHAPE = 8  # 1 for each joint position, plus gripper

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        conv_channels_1 = 32
        conv_channels_2 = 32
        conv_channels_3 = 32

        self.overhead_cam_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.OVERHEAD_CAM_SHAPE[2], conv_channels_1, kernel_size=5)),
            ('bn1', nn.BatchNorm2d(conv_channels_1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(conv_channels_1, conv_channels_2, kernel_size=5)),
            ('bn2', nn.BatchNorm2d(conv_channels_2)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(conv_channels_2, conv_channels_3, kernel_size=5)),
            ('bn3', nn.BatchNorm2d(conv_channels_3)),
            ('relu3', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d()),  # TODO: what param for MaxPool
        ]))

        self.wrist_cam_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.WRIST_CAM_SHAPE_CAM_SHAPE[2], conv_channels_1, kernel_size=5)),
            ('bn1', nn.BatchNorm2d(conv_channels_1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(conv_channels_1, conv_channels_2, kernel_size=5)),
            ('bn2', nn.BatchNorm2d(conv_channels_2)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(conv_channels_2, conv_channels_3, kernel_size=5)),
            ('bn3', nn.BatchNorm2d(conv_channels_3)),
            ('relu3', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d()),  # TODO: what param for MaxPool
        ]))

        # TODO: likely will merge networks in the forward() call
        self.merge_camera_nets = None

        motor_fc_1 = 512
        motor_fc_2 = 512

        self.motor_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.MOTOR_SIGNAL_SHAPE, motor_fc_1)),
            ('dropout1', nn.Dropout()),
            ('fc2', nn.Linear(motor_fc_1, motor_fc_2)),
        ]))

        # Can run the custom weight initialization method here
        # self.apply(self.init_weights)

    def forward(self, x):
        """
        Step the network forward
        """

        # TODO: This convert might not be needed if bug is fixed in test code before this
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
