from agents import *
from envs import *
from utils import *
from gym.wrappers import Monitor
from pommerman import agents
import os

N_CHANNELS = 16


def main():
    n_episodes = 100
    config_id = default_config['ConfigID']

    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]

    env = pommerman.make(config_id, agent_list)
    env.reset()
    output_size = env.action_space.n  # 2

    use_cuda = True
    is_render = False
    model_path = './nv03/models/{}.model'.format(config_id)
    predictor_path = './nv03/models/{}.pred'.format(config_id)
    target_path = './nv03/models/{}.target'.format(config_id)

    gamma = float(default_config['Gamma'])

    agent = RNDAgent(
        N_CHANNELS,
        output_size,
        gamma,
        training=False
    )

    print('Loading Pre-trained model....')
    if use_cuda:
        agent.model.load_state_dict(torch.load(model_path))
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
        agent.rnd.target.load_state_dict(torch.load(target_path))
    else:
        agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
        agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
    print('End load...')

    wins = 0
    losses = 0
    ties = 0
    agent_list = [
        agent,
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.DockerAgent("multiagentlearning/nips19-tu2id4n.hit_mhp_agent_v1", port=12333),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    for i in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            if is_render:
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

            if done:
                result = 'win'
                if info['result'] == constants.Result.Win:
                    if 0 in info['winners']:
                        wins += 1
                        result = 'Win'
                    else:
                        losses += 1
                        result = 'Loss'
                elif info['result'] == constants.Result.Tie:
                    ties += 1
                    result = 'Tie'

                print('Result: {}'.format(result))

    print('winrate: {}'.format(wins / n_episodes))
    print('losses: {}'.format(losses / n_episodes))
    print('tie: {}'.format(ties / n_episodes))


if __name__ == '__main__':
    main()
