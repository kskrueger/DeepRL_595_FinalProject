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

        # TODO: Set *all* the parameters below this line
        img_conv_channels_1 = 32
        img_conv_channels_2 = 32
        img_conv_channels_3 = 32
        img_max_pooling = (2, 2)

        frame_merge_size = 0  # TODO: what is this concat/merged size?
        frame_merge_channels = 32
        frame_merge_dropout = .5
        frame_merge_fc = 512

        motor_fc_1 = 512
        motor_dropout = .5
        motor_fc_2 = 512

        final_merged_size = 0  # TODO: what is concat/merged size?
        final_fc_1 = 512
        final_dropout = .5
        final_fc_2 = 512
        final_out_actions = self.NUM_ACTIONS

        self.overhead_cam_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.OVERHEAD_CAM_SHAPE[2], img_conv_channels_1, kernel_size=(7, 7), stride=(2, 2))),
            ('bn1', nn.BatchNorm2d(img_conv_channels_1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(img_conv_channels_1, img_conv_channels_2, kernel_size=(5, 5))),
            ('bn2', nn.BatchNorm2d(img_conv_channels_2)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(img_conv_channels_2, img_conv_channels_3, kernel_size=(3, 3))),
            ('bn3', nn.BatchNorm2d(img_conv_channels_3)),
            ('relu3', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(img_max_pooling)),
        ]))

        self.wrist_cam_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.WRIST_CAM_SHAPE_CAM_SHAPE[2], img_conv_channels_1, kernel_size=(7, 7), stride=(2, 2))),
            ('bn1', nn.BatchNorm2d(img_conv_channels_1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(img_conv_channels_1, img_conv_channels_2, kernel_size=(5, 5))),
            ('bn2', nn.BatchNorm2d(img_conv_channels_2)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(img_conv_channels_2, img_conv_channels_3, kernel_size=(3, 3))),
            ('bn3', nn.BatchNorm2d(img_conv_channels_3)),
            ('relu3', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(img_max_pooling)),
        ]))

        self.merged_frames_net = self.wrist_cam_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(frame_merge_size, frame_merge_channels, kernel_size=2)),  # TODO: kernel?
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(frame_merge_dropout)),
            ('fc1', nn.Linear(frame_merge_channels, frame_merge_fc)),
            ('relu2', nn.ReLU()),
        ]))

        self.motor_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.MOTOR_SIGNAL_SHAPE, motor_fc_1)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(motor_dropout)),
            ('fc2', nn.Linear(motor_fc_1, motor_fc_2)),
            ('relu2', nn.ReLU()),
        ]))

        self.final_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(final_merged_size, final_fc_1)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(final_dropout)),
            ('fc2', nn.Linear(final_fc_1, final_fc_2)),
            ('relu2', nn.ReLU()),
            ('output', nn.Linear(final_fc_2, final_out_actions)),
        ]))

        # Can run a custom weight initialization method here
        # self.apply(self.init_weights)

    def forward(self, overhead_frame_x, wrist_frame_x, motor_signals_x):
        """
        Step the network forward
        """

        # TODO: This convert might not be needed if bug is fixed in test code before this
        # if type(x) is np.ndarray:
        #     x = np.array([x])
        #     x = x.swapaxes(1, 3).astype(np.float32)
        #     x = torch.from_numpy(x)

        overhead_frame_x = overhead_frame_x.to(self.device)
        wrist_frame_x = wrist_frame_x.to(self.device)
        motor_signals_x = motor_signals_x.to(self.device)

        overhead_out = self.overhead_cam_net.forward(overhead_frame_x)
        wrist_out = self.wrist_cam_net.forward(wrist_frame_x)

        merge1_x = torch.cat([overhead_out, wrist_out], 1)

        merge1_out = self.merged_frames_net(merge1_x)

        motor_out = self.motor_net(motor_signals_x)

        merge2_x = torch.cat([merge1_out, motor_out], 1)

        final_out = self.final_net(merge2_x)

        return final_out

    # Can add a custom weight initialization method here
    def init_weights(self, m):
        print("Initialized weights to custom method")
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.uniform(m.weight, -.01, .01)
            m.bias.data.fill_(.01)
