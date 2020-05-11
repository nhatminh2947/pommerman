"""
A Work-In-Progress navocado using Tensorforce
"""
import os
import json
from pommerman.agents import BaseAgent
from pommerman import characters, constants, envs
import numpy as np
from tqdm import tqdm
from navocado import env_wrap
import time

from collections import defaultdict
import queue

import gym
import ray
from ray.rllib.agents.a3c.a2c import A2CAgent
from ray.rllib.agents.a3c.a3c_tf_policy_graph import A3CPolicyGraph
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

VERY_SMALL_FLOAT = -1e6

import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.visionnet import VisionNetwork


class VisionActMaskNetwork(VisionNetwork):
    def _build_layers(self, inputs, num_outputs, options):
        frame_input, action_mask, step_mask = inputs[:, :, :, :18], inputs[:, :, :, 18], inputs[:, :, :, 19]
        layer_1, layer_2 = super()._build_layers(frame_input,
                                                 num_outputs,
                                                 options['custom_options'])
        step_mask = tf.concat([tf.reshape(step_mask, [-1, 121]),
                               step_mask[:, 0, 0][:, None]], axis=1)
        if num_outputs == 1:
            layer_2 = layer_2 * step_mask[:, 0]
            return layer_1, layer_2
        # layer_1 = tf.Print(layer_1, [tf.shape(layer_1)], "layer_1 ")
        # layer_2 = tf.Print(layer_2, [tf.shape(layer_2)], "layer_2 ")
        bomb_action_mask = tf.cast(inputs[:, 0, 0, 12] > 0, tf.float32)[:, None]
        # bomb_action_mask = tf.tile(tf.constant(1.0)[None, None],
        #                           [tf.shape(action_mask)[0], 1])
        action_mask = tf.concat([tf.reshape(action_mask, [-1, 121]),
                                 bomb_action_mask], axis=1)
        layer_1 = layer_1 * step_mask
        layer_1 = layer_1 + (1.0 - action_mask) * VERY_SMALL_FLOAT
        return layer_1, layer_2


class RLLibV2Agent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""

    def __init__(self, model_paths=None, can_kick=False, character=characters.Bomber):
        super(RLLibV2Agent, self).__init__(character)
        self.agents = []
        self.model_paths = model_paths
        self.env = None
        self.trainable = True
        self._action_dist = {x: 0 for x in range(6)}
        self.agent_id_str = None
        self._can_kick = can_kick
        # print(envs.env_wrap.PommermanEnv)

    def initialize(self, env):
        self.env = env_wrap.PommermanEnv(env, use_kick=self._can_kick)
        register_env("pommerman_env_fake", env_wrap.env_creator)

        fake_obs = self.env._env.reset()[0]

        for idx, model_path in enumerate(self.model_paths):
            if not os.path.isfile(model_path + '.extra_data'):
                print('[Error] File {} not found.' % model_path)
            self.agents.append(self._initialize_one(idx, model_path))

        # preprocess
        st_time = time.time()
        self.act_prepare(fake_obs, [])
        ed_time = time.time()
        print('Preprocess use %ss.' % (ed_time - st_time))

        return self.agents

    def _initialize_one(self, idx, model_path):
        obs_space = gym.spaces.Box(0, 20, shape=[11, 11, 20])
        act_space = gym.spaces.Discrete(122)

        policy_graphs = {
            "a2c_policy": (A3CPolicyGraph, obs_space, act_space, {}),
            "random_policy": (A3CPolicyGraph, obs_space, act_space, {})
        }

        ModelCatalog.register_custom_model("act_mask_model", VisionActMaskNetwork)

        def policy_mapping_fn(agent_id):
            if agent_id == 'P0' or agent_id == 'P2':
                return "a2c_policy"
            else:
                return "random_policy"

        agent = A2CAgent(
            env="pommerman_env_fake",
            config={
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": policy_mapping_fn,
                    "policies_to_train": ["a2c_policy"],
                },
                "observation_filter": "NoFilter",
                "num_workers": 0,
                "vf_loss_coeff": 0.5,
                "entropy_coeff": -0.01,
                "gamma": 0.99,
                "grad_clip": 40.0,
                "lambda": 1.0,
                "lr": 0.0001,
                "model": {
                    "custom_model": "act_mask_model",
                    "custom_options": {
                        "dim": 11,
                        "conv_filters": [[16, [3, 3], 1],
                                         [32, [3, 3], 1],
                                         [64, [3, 3], 1],
                                         [64, [11, 11], 1]]
                    }
                }
            })

        agent.restore(model_path)

        return agent

    def act_prepare(self, obs, action_space):
        direct_act = self.act_core(obs, action_space, 10.0)
        self.agent_id_str = None
        return direct_act

    def act(self, obs, action_space):
        return self.act_core(obs, action_space, 0.08)

    def act_core(self, obs, action_space, limit_time):
        st_time = time.time()

        if not self.agent_id_str:
            agent_num = obs['board'][obs['position']]
            self.agent_id_str = 'P%s' % (agent_num - 10)

        use_bomb = True
        if obs['step_count'] >= 800:
            use_bomb = False

        P0_obs = self.env._normalize_state(obs)
        P0_state = self.env._process_state(P0_obs, use_bomb)

        dest_acts = {}
        direct_acts = [0] * 6
        for idx, agent in enumerate(self.agents):
            P0_dest_act = agent.compute_action(P0_state, policy_id='a2c_policy')
            P0_direct_act = self.env._process_action(P0_obs, P0_dest_act, self.agent_id_str)
            dest_act = self.env._mapping_dest_action(P0_dest_act, self.agent_id_str)
            direct_act = self.env._mapping_direct_action(P0_direct_act, self.agent_id_str)
            dest_acts[direct_act] = dest_act
            direct_acts[direct_act] += 1
            ed_time = time.time()
            if ed_time - st_time > limit_time:
                print('[Warning] Only Ensemble %s Agents.' % idx)
                break

        direct_act = np.argmax(direct_acts).item()
        self.dest_act = dest_acts[direct_act]

        ed_time = time.time()
        if ed_time - st_time > 0.1:
            print("--- %s seconds ---" % (ed_time - st_time))

        return direct_act

    def init_agent(self, id, game_type):
        try:
            super(RLLibV2Agent, self).init_agent(id, game_type)
        except:
            pass
        self.agent_id_str = None
        print('## NavocadoAgent -> Agent', id)

    def episode_end(self, reward):
        self.agent_id_str = None
        print('## NavocadoAgent: Reward', reward)
