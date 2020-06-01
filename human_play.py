from pommerman import agents
import pommerman
from agents import *
from envs import *
from utils import *
import gym

N_CHANNELS = 16


class RewardShaping(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.alive_agents = [0, 1, 2, 3]
        self.training_agent = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.reward(self.alive_agents, obs[0], info)
        self.alive_agents = obs[0]['alive']
        print(self.alive_agents)
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.alive_agents = [0, 1, 2, 3]
        return self.env.reset(**kwargs)

    def reward(self, alive_agents, obs, info):
        reward = 0
        for id in range(10, 14):
            if id in alive_agents and id not in obs['alive']:
                print(id)
                print(obs['enemies'])
                print(constants.Item(value=id) in obs['enemies'])

                if constants.Item(value=id) in obs['enemies']:
                    reward += 0.5
                    print('Enemy killed')
                elif id - 10 == self.training_agent:
                    reward += -1
                    print('Dead')
                elif constants.Item(value=id) == obs['teammate']:
                    reward += -0.5

        if info['result'] == constants.Result.Tie:
            reward += -1

        return reward


def main():
    env_id = 'PommeTeam-v0'

    agent_list = [
        agents.PlayerAgent(),
        StaticAgent(),
        StaticAgent(),
        StaticAgent()
    ]
    env = RewardShaping(pommerman.make(env_id, agent_list))
    env.reset()
    obs = env.reset()
    while True:
        env.render()
        actions = env.act(obs)
        obs, reward, done, info = env.step(actions)
        print(obs[0]['alive'])
        print(reward)
        if done:
            print('info: ', info)
            break


if __name__ == '__main__':
    main()
