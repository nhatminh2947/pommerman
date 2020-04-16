from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pickle

N_CHANNELS = 15

def main():
    print({section: dict(config[section]) for section in config.sections()})
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    agent_list = [
        helpers.make_agent_from_string(agent_string, agent_id)
        for agent_id, agent_string in enumerate(default_config['Agents'].split(','))
    ]
    env = pommerman.make(env_id, agent_list)

    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

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
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])

    sticky_action = False
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    agent = RNDAgent
    env_type = PommeEnvironment

    agent = agent(
        N_CHANNELS,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
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

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id=env_id,
                        agent_list=default_config['Agents'],
                        is_render=False,
                        env_idx=idx,
                        child_conn=child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE])
    for i, work in enumerate(works):
        obs = work.reset()
        states[i, :, :, :] = work.featurize(obs[0])

    steps = 0
    rall = 0
    rd = False
    intrinsic_reward_list = []
    while not rd:
        steps += 1
        actions, value_ext, value_int, policy = agent.get_action(np.float32(states))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_obs, rewards, dones, episode_rewards = [], [], [], []
        for parent_conn in parent_conns:
            obs, reward, episode_reward, done, info = parent_conn.recv()

            next_obs.append(obs)
            rewards.append(reward)
            dones.append(done)

        # print(next_obs)
        # print(np.shape(next_obs))
        next_obs = np.stack(next_obs)

        # total reward = int reward + ext Reward
        intrinsic_reward = agent.compute_intrinsic_reward(next_obs)
        intrinsic_reward_list.append(intrinsic_reward)
        states = next_obs[:, :, :, :]

        if rd:
            intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(
                intrinsic_reward_list)
            with open('int_reward', 'wb') as f:
                pickle.dump(intrinsic_reward_list, f)
            steps = 0
            rall = 0


if __name__ == '__main__':
    main()
