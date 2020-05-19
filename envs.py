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
    def __init__(self, env, training_agents):
        super().__init__(env)
        self.env = env
        self.training_agents = training_agents
        self.steps = 0
        self.episode_reward = 0
        self.num_bombs = 0
        self.alive_agents = [0, 1, 2, 3]

        self.ability = Ability()
        self.env.reset()

    def reward_shaping(self, new_obs, prev_board):
        reward = 0
        current_alive_agents = np.asarray(new_obs['alive']) - constants.Item.Agent0.value
        enemies = [enemy.value - constants.Item.Agent0.value for enemy in new_obs['enemies']]

        if self.training_agents not in current_alive_agents:
            return -1

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

    def step(self, action):
        obs = self.env.get_observations()
        actions = self.env.act(self.env.get_observations())
        actions[self.training_agents] = action
        new_obs, reward, done, info = self.env.step(actions)

        if action == constants.Action.Bomb.value:
            self.num_bombs += 1

        reward = self.reward_shaping(new_obs[self.training_agents], obs[self.training_agents]['board'])
        self.episode_reward += reward
        self.steps += 1

        current_alive_agents = np.asarray(new_obs[self.training_agents]['alive']) - constants.Item.Agent0.value
        if self.training_agents not in current_alive_agents:
            done = True

        if done:
            if self.training_agents not in current_alive_agents:
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

        return new_obs, self.observation(new_obs[self.training_agents]), reward, done, info

    @staticmethod
    def observation(obs):
        id = 0
        features = np.zeros(shape=(16, 11, 11))
        # print(obs)
        for item in constants.Item:
            if item in [constants.Item.Bomb,
                        constants.Item.Flames,
                        constants.Item.Agent0,
                        constants.Item.Agent1,
                        constants.Item.Agent2,
                        constants.Item.Agent3,
                        constants.Item.AgentDummy]:
                continue
            # print("item:", item)
            # print("board:", obs["board"])

            features[id, :, :][obs["board"] == item.value] = 1
            id += 1

        for feature in ["flame_life", "bomb_life", "bomb_blast_strength"]:
            features[id, :, :] = obs[feature]
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

    def reset(self):
        self.steps = 0
        self.episode_reward = 0
        self.alive_agents = [0, 1, 2, 3]
        self.num_bombs = 0
        self.ability.reset()
        observations = self.env.reset()

        return self.observation(observations[self.training_agents])


class PommeEnvironment(Process):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            json_dir=None,
            is_team=False
    ):
        super(PommeEnvironment, self).__init__()
        print(env_id)

        agent_list = []
        if is_team:
            self.training_agents = [(env_idx % 4), ((env_idx % 4) + 2) % 4]  # Agents id is [0, 2] or [1, 3]
        else:
            self.training_agents = env_idx % 4  # Setting for single agent (FFA)

            for i in range(4):
                if i == self.training_agents:
                    agent_list.append(agents.RandomAgent())
                else:
                    agent_list.append(agents.SimpleAgent())

        self.env = pommerman.make(env_id, agent_list)
        self.env = PommeWrapper(self.env, self.training_agents)

        self.is_render = is_render
        self.env_idx = env_idx

        self.episode = 0
        self.child_conn = child_conn
        self.json_dir = json_dir

        self.env.reset()

        print("Training Agents:", self.training_agents)

    def run(self):
        super(PommeEnvironment, self).run()
        while True:
            agent_action = self.child_conn.recv()

            if self.is_render:
                self.env.render()

            raw_obs, obs, reward, done, info = self.env.step(agent_action)

            if done:
                # print(
                #     "Env #{:>2} Episode #{:6} Steps: {:3} Reward: {: 4.4f}\tResult: {}".format(self.env_idx,
                #                                                                                self.episode,
                #                                                                                info['steps'],
                #                                                                                info['episode_reward'],
                #                                                                                info['episode_result']))
                obs = self.reset()

            self.child_conn.send([obs, reward, done, info])

    def reset(self):
        self.episode += 1

        return self.env.reset()
