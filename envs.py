import pommerman
from pommerman import agents
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

        if is_team:
            self.training_agents = [(env_idx % 4), ((env_idx % 4) + 2) % 4]  # Agents id is [0, 2] or [1, 3]
        else:
            self.training_agents = env_idx % 4  # Setting for single agent (FFA)
            agent_list[self.training_agents] = agents.RandomAgent()

        self.env = pommerman.make(env_id, agent_list, '000.json')

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

            self.child_conn.send(
                [utils.featurize(observations[self.training_agents]), reward[self.training_agents], self.episode_reward,
                 done, info])

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.episode_reward = 0

        return self.env.reset()
