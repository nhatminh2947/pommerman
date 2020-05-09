import pommerman
from pommerman import constants, agents
from torch.multiprocessing import Process
import gym
import utils
from agents import StaticAgent
from config import *
import numpy as np
from pommerman import utility


class Ability:
    def __init__(self):
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False

    def reset(self):
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False


class PommeWrapper(gym.Wrapper):
    def __init__(self, env, training_agent):
        super().__init__(env)
        self.env = env
        self.steps = 0
        self.episode_reward = 0
        self.num_bombs = 0
        self.alive_agents = np.arange(4)
        self.training_agent = training_agent

        self.ability = Ability()
        self.env.reset()

    def reward_shaping(self, new_obs, prev_board):
        reward = 0
        current_alive_agents = np.asarray(new_obs['alive']) - constants.Item.Agent0.value
        enemies = [enemy.value - constants.Item.Agent0.value for enemy in new_obs['enemies']]

        if utility.position_is_powerup(prev_board, new_obs['position']):
            if constants.Item(prev_board[new_obs['position']]) == constants.Item.IncrRange:
                reward += 0.01
                self.ability.blast_strength += 1
            elif constants.Item(prev_board[new_obs['position']]) == constants.Item.ExtraBomb:
                reward += 0.01
                self.ability.ammo += 1
            elif not self.ability.can_kick and constants.Item(prev_board[new_obs['position']]) == constants.Item.Kick:
                reward += 0.05
                self.ability.can_kick = True

        for enemy in enemies:
            if enemy in self.alive_agents and enemy not in current_alive_agents:
                reward += 0.5
        self.alive_agents = current_alive_agents

        return reward

    def step(self, actions):
        prev_obs = self.env.get_observations()
        observations, reward, done, info = self.env.step(actions)

        if actions[self.training_agent] == constants.Action.Bomb.value:
            self.num_bombs += 1

        reward = self.reward_shaping(observations[self.training_agent], prev_obs[self.training_agent]['board'])
        self.episode_reward += reward
        self.steps += 1

        current_alive_agents = np.asarray(observations[self.training_agent]['alive']) - constants.Item.Agent0.value
        if self.training_agent not in current_alive_agents:
            done = True

        if done:
            if self.training_agent not in current_alive_agents:
                result = constants.Result.Loss
            elif info['result'] == constants.Result.Win:
                result = constants.Result.Win
            else:
                result = constants.Result.Tie

            info['episode_reward'] = self.episode_reward
            info['episode_result'] = result
            info['num_bombs'] = self.num_bombs
            info['steps'] = self.steps
            info['ammo'] = self.ability.ammo
            info['blast_strength'] = self.ability.blast_strength
            info['can_kick'] = self.ability.can_kick

        return observations, reward, done, info

    def reset(self):
        self.steps = 0
        self.episode_reward = 0
        self.alive_agents = [0, 1, 2, 3]
        self.num_bombs = 0
        self.ability.reset()

        return self.env.reset()


class PommeEnvironment(Process):
    def __init__(
            self,
            env_name,
            is_render,
            env_idx,
            child_conn,
            json_dir=None
    ):
        super(PommeEnvironment, self).__init__()
        print(env_name)

        agent_list = [
            agents.BaseAgent(),
            agents.BaseAgent(),
            agents.BaseAgent(),
            agents.BaseAgent()
        ]

        self.training_agent = env_idx % 4

        self.env = pommerman.make(env_name, agent_list)
        self.env = PommeWrapper(self.env, self.training_agent)

        self.is_render = is_render
        self.env_idx = env_idx

        self.episode = 0
        self.child_conn = child_conn
        self.json_dir = json_dir

        self.env.reset()

        print("Training Agents:", self.training_agent)

    def run(self):
        super(PommeEnvironment, self).run()
        while True:
            agent_actions = self.child_conn.recv()

            if self.is_render:
                self.env.render()

            raw_obs, obs, reward, done, info = self.env.step(agent_actions)

            if done:
                obs = self.reset()

            self.child_conn.send([obs, reward, done, info])

    def reset(self):
        self.episode += 1

        return self.env.reset()
