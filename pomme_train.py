from collections import deque

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
    global_episode = 0

    episode_rewards = deque(maxlen=1000)
    count_bomb = 0
    episode_wins = 0
    episode_ties = 0
    episode_losses = 0
    episode_steps = 0
    episode_this_update = 0

    states = np.zeros([num_worker, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE])

    for i, work in enumerate(works):
        obs = work.reset()
        states[i, :, :, :] = featurize(obs[0])

    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, \
        total_ext_values, total_int_values, total_policy, total_policy_np = \
            [], [], [], [], [], [], [], [], [], [], []

        global_step += (num_worker * num_step)
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(states)  # Normalize state?

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_obs, rewards, dones = [], [], []
            for parent_conn in parent_conns:
                obs, reward, done, info = parent_conn.recv()

                next_obs.append(obs)
                rewards.append(reward)
                dones.append(done)

                if done:
                    episode_rewards.append(info['episode_reward'])
                    count_bomb += info['num_bombs']
                    episode_steps += info['steps']

                    if info['episode_result'] == constants.Result.Win:
                        episode_wins += 1
                    elif info['episode_result'] == constants.Result.Tie:
                        episode_ties += 1
                    else:
                        episode_losses += 1

                    episode_this_update += 1
                    global_episode += 1

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
        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape(
            [-1, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE])
        total_reward = np.stack(total_reward).transpose()
        total_action = np.stack(total_action).transpose().reshape([-1])
        # print(total_action)
        # print(total_action.shape)
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape(
            [-1, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE])
        # print('total_ext_values.shape', np.shape(total_ext_values))
        total_ext_values = np.squeeze(total_ext_values, axis=-1).transpose()
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

        # print('total_int_reward', total_int_reward)
        # print('total_int_values', total_int_values)

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

        # print('int_target: ', int_target)

        # add ext adv and int adv
        total_adv = int_adv * int_coef + ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------
        # print(np.shape(total_int_values[:, :-1]))
        # print(np.shape(int_target))
        # print(np.shape(total_ext_values))
        # print(np.shape(ext_target))

        # Step 5. Training!
        loss, critic_ext_loss, critic_int_loss, actor_loss, forward_loss, entropy = agent.train_model(
            s_batch=total_state,
            target_ext_batch=ext_target,
            target_int_batch=int_target,
            y_batch=total_action,
            adv_batch=total_adv,
            next_obs_batch=total_next_obs,
            old_policy=total_policy)
        # print('episode_rewards', episode_rewards)
        if global_update % 10 == 0 or global_update == 1:
            writer.add_scalar('loss/total_loss', loss, global_update)
            writer.add_scalar('loss/critic_ext_loss', critic_ext_loss, global_update)
            writer.add_scalar('loss/critic_int_loss', critic_int_loss, global_update)
            writer.add_scalar('loss/actor_loss', actor_loss, global_update)
            writer.add_scalar('loss/forward_loss', forward_loss, global_update)
            writer.add_scalar('loss/entropy', entropy, global_update)

            writer.add_scalar('reward/intrinsic_reward', np.sum(total_int_reward) / num_worker, global_update)
            writer.add_scalar('reward/mean_extrinsic_reward', 0 if not episode_rewards else np.mean(episode_rewards),
                              global_update)
            writer.add_scalar('reward/max_extrinsic_reward', 0 if not episode_rewards else np.max(episode_rewards),
                              global_update)

            writer.add_scalar('data/mean_steps_per_episode', episode_steps / episode_this_update, global_update)
            writer.add_scalar('data/mean_bomb_per_episode', count_bomb / episode_this_update,
                              global_update)
            writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), global_update)
            writer.add_scalar('data/win_rate', episode_wins / episode_this_update, global_update)
            writer.add_scalar('data/tie_rate', episode_ties / episode_this_update, global_update)
            writer.add_scalar('data/loss_rate', episode_losses / episode_this_update, global_update)

            writer.add_scalar('value/intrinsic_value', np.mean(total_int_values), global_update)
            writer.add_scalar('value/extrinsic_value', np.mean(total_ext_values), global_update)
            writer.add_scalar('value/iv_explained',
                              explained_variance(total_int_values[:, :-1].reshape([-1]), int_target), global_update)
            writer.add_scalar('value/ev_explained',
                              explained_variance(total_ext_values[:, :-1].reshape([-1]), ext_target), global_update)

            episode_rewards.clear()
            episode_steps = 0
            count_bomb = 0
            episode_this_update = 0
            episode_wins = 0
            episode_ties = 0
            episode_losses = 0

        if global_step % (num_worker * num_step * 50) == 0:
            print('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.rnd.predictor.state_dict(), predictor_path)
            torch.save(agent.rnd.target.state_dict(), target_path)


if __name__ == '__main__':
    main()
