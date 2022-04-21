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

    def __init__(self):
        self.pyb = p
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = spaces.Box(np.array([-1]*5), np.array([1]*5))
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))

    def reset(self, **kwargs):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering after we loaded everything
        p.setGravity(0, 0, -10)
        urdfRootPath = pybullet_data.getDataPath()

        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid, i, rest_poses[i])

        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"), basePosition=[0.65, 0, 0])

        state_object = [random.uniform(0.5, 0.8), random.uniform(-0.2, 0.2), 0.05]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0])
        robot_state_obs = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # rendering's back on again

        observation = self.get_observation(state_robot, robot_state_obs)

        return observation

    def step(self, action, ):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]  # [0, 1]
        wrist_twist = action[4]  # wrist rotation rate

        orientation = p.getQuaternionFromEuler([0, -math.pi, wrist_twist])
        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[:7]

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        if state_object[2]>0.45:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        info = state_object
        robot_state_obs = state_robot + state_fingers

        observation = self.get_observation(state_robot, robot_state_obs)

        return observation, reward, done, info

    def get_observation(self, state_robot, robot_state_obs):
        joint_angles = p.getEulerFromQuaternion(p.getLinkState(self.pandaUid, 11)[1])
        overhead_rgbd = self.get_camera_rgb([.5, 0, .5], .7, 90, -90, 0, rgbd=True)

        wrist_cam_angle = np.degrees(joint_angles[2]) - 90
        wrist_cam_offset = .2
        rot_x = np.cos(-np.radians(wrist_cam_angle)) * wrist_cam_offset
        rot_y = np.sin(-np.radians(wrist_cam_angle)) * wrist_cam_offset

        wrist_rgb = self.get_camera_rgb([state_robot[0]+rot_y, state_robot[1]+rot_x, state_robot[2] - .1], .4,
                                        wrist_cam_angle, -90, 0)

        observation = {'robot_state': robot_state_obs, 'overhead_rgbd': overhead_rgbd, 'wrist_rgb': wrist_rgb}
        return observation

    def close(self):
        p.disconnect()

    def get_camera_rgb(self, position, distance=.7, yaw=90, pitch=-70, roll=0, resolution=(720, 960), rgbd=False):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=position,
                                                            distance=distance,
                                                            yaw=yaw,
                                                            pitch=pitch,
                                                            roll=roll,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(resolution[1]) /resolution[0],
                                                     nearVal=0.01,
                                                     farVal=5)
        ret = p.getCameraImage(width=resolution[1],
                                              height=resolution[0],
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)
        (_, _, px, _, _) = ret

        cam_array = np.array(px)
        cam_array = np.reshape(cam_array, (resolution[0], resolution[1], 4))

        channels = 4 if rgbd else 3
        frame_array = cam_array[:, :, :channels]
        frame_array[:, :, :3] = frame_array[:, :, :3].astype(np.uint8)

        return frame_array

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, 3]
        return rgb_array