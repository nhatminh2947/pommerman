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

    is_render = True
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    use_cuda = False

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
    ties = 0
    print("Evaluating...")
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
                result = constants.Result.Loss
                if constants.Item.Agent0 in obs[0]['alive']:
                    if info['result'] == constants.Result.Win:
                        result = constants.Result.Win
                        wins += constants.Result.Win
                    else:
                        result = constants.Result.Tie
                        ties += constants.Result.Tie

                print("Match #{} ended as a {}".format(i, result.name))

    print("Win rate: {}".format(wins / 100))
    print("There are {} won {} tie".format(wins, ties))


if __name__ == '__main__':
    main()
