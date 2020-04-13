import gym
import cv2

import numpy as np
import pommerman
from pommerman import agents
from pommerman import helpers
from abc import abstractmethod
from collections import deque
from copy import copy

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from torch.multiprocessing import Pipe, Process

from model import *
from config import *
from PIL import Image

train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])


class PommeEnvironment(Process):
    def __init__(
            self,
            env_id,
            agent_list,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84,
    ):
        super(PommeEnvironment, self).__init__()
        self.daemon = True
        print(agent_list)
        print(env_id)

        agent_list = [
            helpers.make_agent_from_string(agent_string, agent_id)
            for agent_id, agent_string in enumerate(default_config['Agents'].split(','))
        ]

        self.env = pommerman.make(env_id, agent_list)

        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.last_action = 0

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(PommeEnvironment, self).run()
        while True:
            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()

            obs, reward, done, info = self.env.step(action)
            if self.is_render:
                self.env.render()

            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print(
                    "[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Stage: {} current x:{}   max x:{}".format(
                        self.episode,
                        self.env_idx,
                        self.steps,
                        self.rall,
                        np.mean(self.recent_rlist),
                        info['stage'],
                        info['x_pos'],
                        self.max_pos))

                self.history = self.reset()

            self.child_conn.send([obs, reward, done])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0

        return self.env.reset()
