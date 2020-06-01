import pommerman
from pommerman import constants, agents
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

        print(env_id)

        agent_list = [
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent()
            # helpers.make_agent_from_string(agent_string, agent_id)
            # for agent_id, agent_string in enumerate(default_config['Agents'].split(','))
        ]

        self.training_agent = env_idx % 4
        self.env = pommerman.make(env_id, agent_list)

        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.child_conn = child_conn
        self.json_dir = json_dir

        self.env.reset()
        self.episode_reward = 0
        self.num_bombs = 0
        self.alive_agents = [0, 1, 2, 3]

        print("Training Agent:", self.training_agent)

    def run(self):
        super(PommeEnvironment, self).run()
        while True:
            agent_action = self.child_conn.recv()

            if self.is_render:
                self.env.render()

            actions = self.env.act(self.env.get_observations())

            actions[self.training_agent] = agent_action
            observations, reward, done, info = self.env.step(actions)

            self.steps += 1

            if agent_action == constants.Action.Bomb.value:
                self.num_bombs += 1

            if (self.training_agent + constants.Item.Agent0.value) not in observations[self.training_agent]['alive']:
                done = True

            reward = self.reward(self.alive_agents, observations[self.training_agent], info)
            self.episode_reward += reward

            if done:
                print(
                    "Env #{}\t\tEpisode #{}\t\tSteps: {}\t\tReward: {}\t\tResult: {}".format(self.env_idx, self.episode,
                                                                                             self.steps,
                                                                                             self.episode_reward,
                                                                                             info['result']))
                info['episode_result'] = info['result']
                info['episode_reward'] = self.episode_reward
                info['num_bombs'] = self.num_bombs
                info['steps'] = self.steps

                observations = self.reset()

            self.alive_agents = observations[0]['alive']

            self.child_conn.send(
                [utils.featurize(observations[self.training_agents]), reward, done, info])

    def reward(self, alive_agents, obs, info):
        reward = 0

        for id in range(10, 14):
            if id in alive_agents and id not in obs['alive']:
                if constants.Item(value=id) in obs['enemies']:
                    reward += 0.5
                elif constants.Item(value=id) == obs['teammate']:
                    reward += -0.5
                elif id - 10 == self.training_agent:
                    reward += -1

        if info['result'] == constants.Result.Tie:
            reward += -1

        return reward

    def reset(self):
        self.steps = 0
        self.episode_reward = 0
        self.episode += 1
        self.num_bombs = 0
        self.alive_agents = [0, 1, 2, 3]

        return self.env.reset()
