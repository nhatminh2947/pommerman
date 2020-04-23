import pommerman
from pommerman import constants
from torch.multiprocessing import Process

import utils
from agents import StaticAgent
from config import *

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
            StaticAgent(),
            StaticAgent(),
            StaticAgent(),
            StaticAgent(),
            # helpers.make_agent_from_string(agent_string, agent_id)
            # for agent_id, agent_string in enumerate(default_config['Agents'].split(','))
        ]

        # if is_team:
        #     self.training_agents = [(env_idx % 4), ((env_idx % 4) + 2) % 4]  # Agents id is [0, 2] or [1, 3]
        # else:
        #     self.training_agents = env_idx % 4  # Setting for single agent (FFA)
        #     agent_list[self.training_agents] = agents.RandomAgent()

        self.training_agents = 0

        self.env = pommerman.make(env_id, agent_list, 'long_line.json')

        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.child_conn = child_conn
        self.json_dir = json_dir

        self.env.reset()
        self.episode_reward = 0

        print("Training Agents:", self.training_agents)

    def run(self):
        super(PommeEnvironment, self).run()
        while True:
            agent_action = self.child_conn.recv()

            if self.is_render:
                self.env.render()

            actions = self.env.act(self.env.get_observations())

            actions[self.training_agents] = agent_action
            observations, reward, done, info = self.env.step(actions)

            self.episode_reward += reward[self.training_agents]
            self.steps += 1

            info['episode_reward'] = self.episode_reward

            if done:
                print("Episode #{} Steps: {} Reward: {} Info: {}".format(self.episode, self.steps,
                                                                                         self.episode_reward,
                                                                                         info))
                observations = self.reset()

            self.child_conn.send(
                [utils.featurize(observations[self.training_agents]), reward[self.training_agents], done, info])

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.episode_reward = 0

        return self.env.reset()
