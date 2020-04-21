import utils

from agents import *
from envs import *
from utils import *


def main():
    print({section: dict(config[section]) for section in config.sections()})
    env_id = default_config['EnvID']

    agent_list = [
        StaticAgent(),
        StaticAgent(),
        StaticAgent(),
        StaticAgent()
    ]
    env = pommerman.make(env_id, agent_list, '000.json')
    env.reset()

    input_size = 16
    output_size = env.action_space.n  # 2

    env.close()

    is_render = True
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    use_cuda = False
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = 1

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    clip_grad_norm = float(default_config['ClipGradNorm'])

    gamma = float(default_config['Gamma'])
    agent = RNDAgent

    agent = agent(
        input_size,
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

    obs = env.reset()
    state = torch.from_numpy(utils.featurize(obs[0])).unsqueeze(0).float()

    while True:
        env.render()
        action, int_value, ext_value, policy = agent.get_action(state=state)

        actions = env.act(obs)
        actions[0] = action[0]
        obs, reward, done, info = env.step(actions)

        if done:
            print('info: ', info)
            break


if __name__ == '__main__':
    main()
