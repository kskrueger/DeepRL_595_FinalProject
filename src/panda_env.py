import time

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, wrist_shape=(720, 960, 3), overhead_shape=(720, 960, 4), motor_shape=(6, 1)):
        self.pyb = p
        self.wrist_shape = wrist_shape
        self.overhead_shape = overhead_shape
        self.motor_shape = motor_shape
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])
        self.action_space = spaces.Discrete(8)  # +/- for each 3 axis, rotation, and gripper
        self.discrete_actions_steps = [[-.1, 0, 0, 0, 1],
                                       [.1, 0, 0, 0, 1],
                                       [0, -.1, 0, 0, 1],
                                       [0, .1, 0, 0, 1],
                                       [0, 0, .1, 0, 1],
                                       [0, 0, -.1, 0, 1],
                                       # [0, 0, 0, .1, 1],
                                       # [0, 0, 0, -.1, 1],
                                       [0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0]]
        self.observation_space = {'wrist': spaces.Box(np.zeros(wrist_shape), np.ones(wrist_shape)),
                                  'overhead': spaces.Box(np.zeros(overhead_shape), np.ones(overhead_shape)),
                                  'motors': spaces.Box(np.zeros(motor_shape), np.ones(motor_shape))}  # TODO: fix this?

    def reset(self, **kwargs):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering after we loaded everything
        p.setGravity(0, 0, -10)
        urdfRootPath = pybullet_data.getDataPath()

        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])

        reset_fingers = .03
        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid, i, rest_poses[i])

        p.resetJointState(self.pandaUid, 9, reset_fingers)
        p.resetJointState(self.pandaUid, 10, reset_fingers)
        for i in range(10):
            p.stepSimulation()

        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"), basePosition=[0.65, 0, 0])

        state_object = [random.uniform(0.5, 0.8), random.uniform(-0.2, 0.2), 0.05]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])
        state_joint_angles = p.getEulerFromQuaternion(p.getLinkState(self.pandaUid, 11)[1])
        robot_state_obs = np.array([*state_robot, state_joint_angles[2], *state_fingers])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # rendering's back on again

        observation = self.get_observation(state_robot, robot_state_obs)

        return observation

    def step(self, discrete_action_idx, ):
        action = self.discrete_actions_steps[discrete_action_idx]
        # print("action", action)
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        dv = 0.05
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        d_wrist_twist = action[3] * dv  # wrist rotation angle
        fingers = action[4]  # [0, 1]

        current_pose = p.getLinkState(self.pandaUid, 11)
        current_angles = p.getEulerFromQuaternion(current_pose[1])
        current_twist = current_angles[2]
        current_position = current_pose[1]
        # print("The dx = {}, dy = {}, dz = {}".format(dx, dy, dz))
        # print("current_position", current_position)
        orientation = p.getQuaternionFromEuler([0, -math.pi, np.pi/2])
        new_position = [current_position[0] + dx,
                        current_position[1] + dy,
                        current_position[2] + dz]
        # new_position = [.5, 0, .5]
        joint_poses = p.calculateInverseKinematics(self.pandaUid, 11, new_position, orientation)[:7]

        # make the fingers binary
        end = False
        if fingers > .5:
            fingers = .03
        else:
            end = True
            fingers = 0

        # print("fingers(after >.5)", fingers)

        p.setJointMotorControlArray(self.pandaUid, list(range(7)) + [9, 10], p.POSITION_CONTROL,
                                    list(joint_poses) + 2 * [fingers])

        for s_i in range(5 if end else 1):
            p.stepSimulation()
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # if end:
        #     time.sleep(5)

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_joint_angles = p.getEulerFromQuaternion(p.getLinkState(self.pandaUid, 11)[1])
        state_fingers = (p.getJointState(self.pandaUid, 9)[0] / .03, p.getJointState(self.pandaUid, 10)[0] / .03)

        # print("state_fingers", state_fingers)
        # TODO: convert the gripper to be binary (open, closed)

        # TODO: if the robot action closes the gripper, then automatically lift 0.5 meters, then check,
        #   is the object above 0.4m, if so give reward, or otherwise penalty
        #           - If the robot closes gripper when it's within a few cm of the object, then give a small reward.
        #           - Lift object gets big reward -  state_object > 0.45 and the gripper being closed
        #           - Each timestep is small negative reward

        # TODO: add out reward function values here
        LIFT_HEIGHT = .5
        SUCCESS_GRASP_REWARD = 100
        TRIED_FAILED_GRASP_REWARD = min(20, 1 / np.linalg.norm(
            np.array(state_robot[:3]) - np.array(state_object[:3])))  # 30
        TIMESTEP_PENALTY = 0  # -.01
        MIN_TOUCH_DIST = 1
        BAD_GRIP_PENTALY = -5
        MAX_TRIES = 50
        SINGULARITY_PENALTY = -10

        # Give a penalty for every step
        if end:
        # if state_fingers[0] > .8:
            done = True  # end the episode because the fingers closed

            new_position[2] = LIFT_HEIGHT
            joint_poses = p.calculateInverseKinematics(self.pandaUid, 11, new_position, orientation)[:7]
            p.setJointMotorControlArray(self.pandaUid, list(range(7)) + [9, 10], p.POSITION_CONTROL,
                                        list(joint_poses) + 2 * [fingers])

            i = 0
            while True:
                p.stepSimulation()

                # p.setJointMotorControlArray(self.pandaUid, list(range(7)) + [9, 10], p.POSITION_CONTROL,
                #                             list(joint_poses) + 2 * [fingers])

                state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
                state_robot = p.getLinkState(self.pandaUid, 11)[0]
                # print("stateRobot", state_robot)
                if state_robot[2] > LIFT_HEIGHT - .1:
                    break
                if i > MAX_TRIES:
                    break
                i += 1

            if i > MAX_TRIES:
                reward = SINGULARITY_PENALTY
            elif state_object[2] > LIFT_HEIGHT - .2:
                reward = SUCCESS_GRASP_REWARD
            elif np.linalg.norm(np.array(state_robot[:3]) - np.array(state_object[:3])) < MIN_TOUCH_DIST:
                reward = TRIED_FAILED_GRASP_REWARD
            else:
                reward = BAD_GRIP_PENTALY
        else:
            reward = TIMESTEP_PENALTY  # or add step penalty
            done = False

        info = state_object
        robot_state_obs = np.array([*state_robot, state_joint_angles[2], *state_fingers])

        observation = self.get_observation(state_robot, robot_state_obs)

        return observation, reward, done, info

    def get_observation(self, state_robot, robot_state_obs):
        # TODO: use the global OVERHEAD_SHAPE and WRIST_SHAPE passed into these get_camera_frame calls
        # TODO: downsize the frames before returning (should be able to do this already in the above)
        overhead_rgbd = self.get_camera_frames([.5, 0, .5], .7, 90, -90, 0, rgbd=True,
                                               resolution=self.overhead_shape[:2])

        wrist_cam_angle = np.degrees(robot_state_obs[4]) - 90
        wrist_cam_offset = .2
        rot_x = np.cos(-np.radians(wrist_cam_angle)) * wrist_cam_offset
        rot_y = np.sin(-np.radians(wrist_cam_angle)) * wrist_cam_offset

        wrist_rgb = self.get_camera_frames([state_robot[0] + rot_y, state_robot[1] + rot_x, state_robot[2] - .1], .4,
                                           wrist_cam_angle, -90, 0, resolution=self.wrist_shape[:2])

        observation = {'motors': np.atleast_2d(robot_state_obs), 'overhead': overhead_rgbd, 'wrist': wrist_rgb}
        return observation

    def close(self):
        p.disconnect()

    def get_camera_frames(self, position, distance=.7, yaw=90, pitch=-70, roll=0, resolution=(720, 960), rgbd=False,
                          normalize=True):
        near = 0.01
        far = 2.0

        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=position,
                                                          distance=distance,
                                                          yaw=yaw,
                                                          pitch=pitch,
                                                          roll=roll,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(resolution[1]) / resolution[0],
                                                   nearVal=near,
                                                   farVal=far)
        ret = p.getCameraImage(width=resolution[1],
                               height=resolution[0],
                               viewMatrix=view_matrix,
                               projectionMatrix=proj_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        (_, _, img, depth, segmentation) = ret

        depth_array = np.atleast_3d(far * near / (far - (far - near) * depth))
        # depth_img = (depth_array / far * 255).astype(np.uint8)
        img_array = np.array(img, dtype=np.uint8)
        img_array = np.reshape(img_array, (resolution[0], resolution[1], 4))
        # img_array = img_array[:, :, [2, 1, 0, 3]]  # convert to BGR

        frame_array = img_array[:, :, :3]
        if normalize:
            frame_array = frame_array.astype(np.float32) / 255.0
            depth_array /= far

        if not rgbd:
            return frame_array
        else:
            return np.concatenate([frame_array, depth_array], axis=2)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=-70,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, 3]
        return rgb_array
