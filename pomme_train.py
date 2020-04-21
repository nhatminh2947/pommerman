from torch.multiprocessing import Pipe
from torch.utils.tensorboard import SummaryWriter

from agents import *
from envs import *
from utils import *

N_CHANNELS = 16


def main():
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'pomme':
        agent_list = [
            StaticAgent(),
            StaticAgent(),
            StaticAgent(),
            StaticAgent()
            # helpers.make_agent_from_string(agent_string, agent_id)
            # for agent_id, agent_string in enumerate(default_config['Agents'].split(','))
        ]
        env = pommerman.make(env_id, agent_list)
    else:
        raise NotImplementedError

    input_size = env.observation_space.shape
    output_size = env.action_space.n

    print('observation space:', input_size)
    print('action space:', output_size)

    env.close()

    is_load_model = default_config.getboolean('LoadModel')
    is_render = default_config.getboolean('Render')

    # print(is_load_model)

    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    writer = SummaryWriter()

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    json_dir = default_config['JsonDir']

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE))
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma)

    if default_config['EnvType'] == 'pomme':
        env_type = PommeEnvironment
    else:
        raise NotImplementedError

    agent = RNDAgent(
        N_CHANNELS,
        output_size,
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
    )

    if is_load_model:
        print('load model...')
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
            agent.rnd.target.load_state_dict(torch.load(target_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
        print('load finished!')

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id=env_id,
                        agent_list=default_config['Agents'],
                        is_render=is_render,
                        env_idx=idx,
                        child_conn=child_conn,
                        json_dir=json_dir)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    global_update = 0
    global_step = 0

    states = np.zeros([num_worker, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE])

    for i, work in enumerate(works):
        obs = work.reset()
        states[i, :, :, :] = featurize(obs[0])

    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
            [], [], [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(states)  # Normalize state?

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_obs, rewards, dones, episode_rewards = [], [], [], []
            for parent_conn in parent_conns:
                obs, reward, episode_reward, done, info = parent_conn.recv()

                next_obs.append(obs)
                rewards.append(reward)
                dones.append(done)
                episode_rewards.append(episode_reward)

            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            next_obs = np.stack(next_obs)

            # total reward = int reward + ext Reward
            intrinsic_reward = agent.compute_intrinsic_reward(next_obs)
            intrinsic_reward = np.hstack(intrinsic_reward)

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())

            states = next_obs[:, :, :, :]

        # print('states.shape:', states.shape)
        # calculate last next value
        _, value_ext, value_int, _ = agent.get_action(states)  # Normalize state?
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------

        total_state = np.stack(total_state).reshape([-1, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).reshape([-1, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE])
        # print('total_ext_values.shape', np.shape(total_ext_values))
        total_ext_values = np.squeeze(total_ext_values, axis=-1).transpose()
        # print('total_ext_values.shape', np.shape(total_ext_values))
        total_int_values = np.squeeze(total_int_values, axis=-1).transpose()
        total_logging_policy = np.vstack(total_policy_np)

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        # writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        # writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(total_reward,
                                              total_done,
                                              total_ext_values,
                                              gamma,
                                              num_step,
                                              num_worker)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              int_gamma,
                                              num_step,
                                              num_worker)

        # add ext adv and int adv
        total_adv = int_adv * int_coef + ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        agent.train_model(np.float32(total_state), ext_target, int_target, total_action,
                          total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                          total_policy)

        if global_step % (num_worker * num_step * 100) == 0:
            print('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.rnd.predictor.state_dict(), predictor_path)
            torch.save(agent.rnd.target.state_dict(), target_path)


if __name__ == '__main__':
    main()
