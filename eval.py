from agents import *
from envs import *
from utils import *
from gym.wrappers import Monitor
from pommerman import agents

N_CHANNELS = 16


def main():
    print({section: dict(config[section]) for section in config.sections()})
    env_id = default_config['EnvID']

    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent()

    ]

    env = pommerman.make(env_id, agent_list)
    env.reset()

    input_size = 16
    output_size = env.action_space.n  # 2
    print('output_size:', output_size)
    env.close()

    use_cuda = True
    is_render = True
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    gamma = float(default_config['Gamma'])
    agent = RNDAgent

    agent = agent(
        N_CHANNELS,
        output_size,
        gamma
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
    tie = 0

    for i in range(100):
        obs = env.reset()
        state = torch.from_numpy(utils.featurize(obs[0])).unsqueeze(0).float().numpy()
        done = False
        while not done:
            env.render()
            action = agent.act(state)

            actions = env.act(obs)
            actions[0] = action
            obs, reward, done, info = env.step(actions)
            state = torch.from_numpy(utils.featurize(obs[0])).unsqueeze(0).float().numpy()

            if done:
                print('info: ', info)
                if info['result'] == constants.Result.Win:
                    if 0 in info['winners']:
                        wins += 1
                    else:
                        losses += 1
                else:
                    tie += 1

    print('winrate: {}'.format(wins / 100))


if __name__ == '__main__':
    main()
