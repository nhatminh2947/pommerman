import gym
import cv2

import numpy as np
import pommerman
from pommerman import constants
from pommerman import agents
from pommerman import helpers
from pommerman import characters
from abc import abstractmethod
from collections import deque
from copy import copy
import pathlib

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
            json_dir=None,
            is_team=False
    ):
        super(PommeEnvironment, self).__init__()
        self.daemon = True
        print(agent_list)
        print(env_id)

        agent_list = [
            helpers.make_agent_from_string(agent_string, agent_id)
            for agent_id, agent_string in enumerate(default_config['Agents'].split(','))
        ]

        if is_team:
            self.training_agents = [(env_idx % 4), ((env_idx % 4) + 2) % 4]  # Agents id is [0, 2] or [1, 3]
        else:
            self.training_agents = env_idx % 4  # Setting for single agent (FFA)
            agent_list[self.training_agents] = agents.RandomAgent()

        self.env = pommerman.make(env_id, agent_list)

        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.child_conn = child_conn
        self.json_dir = json_dir

        print("Training Agents:", self.training_agents)
        self.current_obs = self.reset()

    def run(self):
        super(PommeEnvironment, self).run()
        while True:
            training_agent_action = self.child_conn.recv()

            if self.is_render:
                self.env.render()

            actions = self.env.act(self.current_obs)

            actions[self.training_agents] = training_agent_action
            observations, reward, done, info = self.env.step(actions)

            self.current_obs = observations

            if not self.env._agents[self.training_agents].is_alive:
                # print(self.training_agents)
                # print(observations[self.training_agents])
                # print(reward)
                done = True

            self.episode_reward += reward[self.training_agents]
            self.steps += 1

            # if self.json_dir is not None:
            #     dir = '{}/env_{}/{}'.format(self.json_dir, self.env_idx, self.episode)
            #     pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
            #     self.env.save_json(dir)

            if done:
                print("[Episode {}({})] Step: {} Episode reward: {}".format(self.episode,
                                                                            self.env_idx,
                                                                            self.steps,
                                                                            self.episode_reward))
                self.current_obs = self.reset()
            training_agent_obs = []

            # for id in self.training_agents:
            #     training_agent_obs.append(self.featurize(observations[id]))
            # print('observation[{}] = {}'.format(self.training_agents, observations[self.training_agents]['board']))
            #
            self.child_conn.send(
                [self.featurize(observations[self.training_agents]), reward[self.training_agents], self.episode_reward,
                 done, info])

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.episode_reward = 0

        return self.env.reset()

    def featurize(self, obs):
        # print(obs)
        id = 0
        features = np.zeros(shape=(15, 11, 11))
        # print(features)
        for item in [constants.Item.Passage,
                     constants.Item.Rigid,
                     constants.Item.Wood,
                     constants.Item.ExtraBomb,
                     constants.Item.IncrRange,
                     constants.Item.Kick]:
            features[id, :, :][obs["board"] == item.value] = 1
            id += 1
        # print(id)
        features[id, :, :] = obs["flame_life"]
        id += 1

        features[id, :, :] = obs["bomb_life"]
        id += 1

        features[id, :, :] = obs["bomb_blast_strength"]
        id += 1

        features[id, :, :][obs["position"]] = 1
        id += 1

        features[id, :, :][obs["board"] == obs["teammate"].value] = 1
        id += 1

        for enemy in obs["enemies"]:
            features[id, :, :][obs["board"] == enemy.value] = 1
        id += 1

        features[id, :, :] = np.full(shape=(11, 11), fill_value=obs["ammo"])
        id += 1

        features[id, :, :] = np.full(shape=(11, 11), fill_value=obs["blast_strength"])
        id += 1

        features[id, :, :] = np.full(shape=(11, 11), fill_value=(1 if obs["can_kick"] else 0))
        id += 1

        return features
