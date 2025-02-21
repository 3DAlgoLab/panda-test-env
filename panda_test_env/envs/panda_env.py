# https://www.etedal.net/2020/04/pybullet-panda_2.html

import math
import os
import random

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import error, spaces, utils
from gym.utils import seeding




class PandaEnv(gym.Env):
    metadata = {"render.models": ["human"]}

    def __init__(self):
        self.step_counter = 0
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-35,  # -40
            cameraTargetPosition=[0.55, -0.35, 0.2],
        )

        # coordinate of end effector and grab
        self.action_space = spaces.Box(np.array([-1] * 4), np.array([1] * 4))
        # joint variables of each fingers
        self.observation_space = spaces.Box(np.array([-1] * 5), np.array([1] * 5))

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.0, -math.pi, math.pi / 2.0])
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.panda_uid, 11)
        currentPosition = currentPose[0]
        newPosition = [
            currentPosition[0] + dx,
            currentPosition[1] + dy,
            currentPosition[2] + dz,
        ]
        jointPoses = p.calculateInverseKinematics(
            self.panda_uid, 11, newPosition, orientation
        )[:7]

        p.setJointMotorControlArray(
            self.panda_uid,
            list(range(7)) + [9, 10],
            p.POSITION_CONTROL,
            list(jointPoses) + 2 * [fingers],
        )

        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.object_uid)
        state_robot = p.getLinkState(self.panda_uid, 11)[0]
        state_fingers = (
            p.getJointState(self.panda_uid, 9)[0],
            p.getJointState(self.panda_uid, 10)[0],
        )

        # Calc Reward
        if state_object[2] > 0.45:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        info = {"object_position": state_object}
        observation = state_robot + state_fingers
        return np.array(observation).astype(np.float32), reward, done, info

    def reset(self):
        p.resetSimulation()
        # disable rendering temporarily
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.81)
        urdf_root_path = pybullet_data.getDataPath()
        p.loadURDF(
            os.path.join(urdf_root_path, "plane.urdf"),
            basePosition=[0, 0, -0.65],
        )
        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]

        self.panda_uid = p.loadURDF(
            os.path.join(urdf_root_path, "franka_panda/panda.urdf"),
            useFixedBase=True,
        )
        for i in range(7):
            p.resetJointState(self.panda_uid, i, rest_poses[i])

        p.loadURDF(
            os.path.join(urdf_root_path, "table/table.urdf"),
            basePosition=[0.5, 0, -0.65],
        )

        p.loadURDF(
            os.path.join(urdf_root_path, "tray/traybox.urdf"),
            basePosition=[0.65, 0, 0],
        )

        state_object = [
            random.uniform(0.5, 0.8),
            random.uniform(-0.2, 0.2),
            0.05,
        ]
        self.object_uid = p.loadURDF(
            os.path.join(urdf_root_path, "random_urdfs/000/000.urdf"),
            basePosition=state_object,
        )

        state_robot = p.getLinkState(self.panda_uid, 11)[0]
        state_fingers = (
            p.getJointState(self.panda_uid, 9)[0],  # 0: joint position
            p.getJointState(self.panda_uid, 10)[0],  # disable for experiment
        )
        observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # rendering's back on again
        return observation

    def render(self, mode="human"):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.7, 0, 0.05],
            distance=0.7,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2,
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(960) / 720, nearVal=0.1, farVal=100.0
        )

        (_, _, px, _, _) = p.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = PandaEnv()
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
