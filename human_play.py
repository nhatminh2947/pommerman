from pommerman import agents
import pommerman
from agents import *
from envs import *
from utils import *

N_CHANNELS = 16


def main():
    env_id = default_config['EnvID']

    agent_list = [
        agents.PlayerAgent(),
        StaticAgent(),
        StaticAgent(),
        StaticAgent()
    ]
    env = pommerman.make(env_id, agent_list, 'a_line.json')
    env.reset()
    obs = env.reset()
    while True:
        env.render()
        actions = env.act(obs)
        obs, reward, done, info = env.step(actions)
        print(obs[0]['board'])
        if done:
            print('info: ', info)
            break


if __name__ == '__main__':
    main()