from collections import deque

from torch.multiprocessing import Pipe
from torch.utils.tensorboard import SummaryWriter
from pommerman import agents
from agents import *
from envs import *
from utils import *
from sample_batch import SampleBatch

N_CHANNELS = 16


def main():
    config_id = default_config['ConfigID']
    env_type = default_config['EnvType']

    if env_type == 'pomme':
        agent_list = [
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent()
        ]
        env = pommerman.make(config_id, agent_list)
    else:
        raise NotImplementedError

    output_size = env.action_space.n

    env.close()

    is_load_model = default_config.getboolean('LoadModel')
    is_render = default_config.getboolean('Render')

    model_path = 'models/{}.model'.format(config_id)
    predictor_path = 'models/{}.pred'.format(config_id)
    target_path = 'models/{}.target'.format(config_id)

    writer = SummaryWriter(filename_suffix='FFA_SimpleAgent')

    logging_interval = int(default_config['LoggingInterval'])

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')

    json_dir = default_config['JsonDir']

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])
    max_updates = int(default_config['MaxUpdates'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])

    reward_rms = RunningMeanStd()
    # obs_rms = RunningMeanStd(shape=(1, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE))
    # pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma)

    n_agents = 2

    agent_pool = [RNDAgent(
        input_size=N_CHANNELS,
        output_size=output_size,
        gamma=gamma,
        training=False,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
    )] * n_agents

    # agent_pool += [agents.SimpleAgent(), agents.SimpleAgent()]

    training_agent = RNDAgent(
        input_size=N_CHANNELS,
        output_size=output_size,
        gamma=gamma,
        training=False,
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

    enemy = agents.DockerAgent('multiagentlearning/navocado', port=12345)

    if is_load_model:
        print('load model...')
        for training_agent in agent_pool:
            if not isinstance(training_agent, RNDAgent):
                continue
            if use_cuda:
                training_agent.model.load_state_dict(torch.load(model_path))
                training_agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
                training_agent.rnd.target.load_state_dict(torch.load(target_path))
            else:
                training_agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                training_agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
                training_agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
        print('load finished!')

    env_type = PommeEnvironment
    workers = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        worker = env_type(env_name=config_id,
                          is_render=is_render,
                          env_idx=idx,
                          child_conn=child_conn,
                          json_dir=json_dir)
        worker.start()
        workers.append(worker)
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
    episode_agent_ammo = 0
    episode_agent_blast_strength = 0
    episode_agent_can_kick = 0

    # states = np.zeros([num_worker, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE])

    observations = []

    for i, worker in enumerate(workers):
        obs = worker.reset()
        observations.append(obs)
        # states[i, :, :, :] = obs[0]

    observations = np.asarray(observations)

    sample_batch = SampleBatch()

    agent_list = [
        training_agent,
        enemy,
        enemy,
        enemy
    ]

    while global_update < max_updates:
        sample_batch.reset()

        global_step += (num_worker * num_step)
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(num_step):
            agent_actions, value_exts, value_ints, policies = [], [], [], []

            for observation, parent_conn in zip(observations, parent_conns):
                actions = []
                for i, (agent, obs) in enumerate(zip(agent_list, observation)):
                    if i == 0:
                        action, value_ext, value_int, policy = agent.get_action(obs)

                        actions.append(action)
                        agent_actions.append(action)
                        value_ints.append(value_int)
                        value_exts.append(value_ext)
                        policies.append(policy)
                    else:
                        actions.append(agent.act(obs, None))

                parent_conn.send(actions)

            next_obs, rewards, intrinsic_rewards, dones = [], [], [], []
            for parent_conn in parent_conns:
                obs, reward, done, info = parent_conn.recv()

                next_obs.append(obs)
                rewards.append(reward[0])
                dones.append(done)

                if done:
                    episode_rewards.append(info['episode_reward'])
                    count_bomb += info['num_bombs']
                    episode_steps += info['steps']
                    episode_agent_ammo += info['ammo']
                    episode_agent_blast_strength += info['blast_strength']
                    episode_agent_can_kick += info['can_kick']

                    if info['episode_result'] == constants.Result.Win:
                        episode_wins += 1
                    elif info['episode_result'] == constants.Result.Tie:
                        episode_ties += 1
                    else:
                        episode_losses += 1

                    episode_this_update += 1
                    global_episode += 1

            next_obs = np.asarray(next_obs)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)

            # total reward = int reward + ext Reward
            # print(next_obs)
            intrinsic_rewards = training_agent.compute_intrinsic_reward(next_obs[:, 0])
            intrinsic_rewards = np.hstack(intrinsic_rewards)

            sample_batch.add(next_obs=next_obs[:, 0],
                             intrinsic_reward=intrinsic_rewards,
                             states=observations[:, 0],
                             rewards=rewards,
                             dones=dones,
                             actions=agent_actions,
                             value_ext=value_exts,
                             value_int=value_ints,
                             policy=policies)

            observations = next_obs

        sample_batch.preprocess()

        # Step 5. Training!
        loss, critic_ext_loss, critic_int_loss, actor_loss, forward_loss, entropy = training_agent.train_model(
            sample_batch)

        if global_update % logging_interval == 0 or global_update == 1:
            writer.add_scalar('loss/total_loss', loss, global_update)
            writer.add_scalar('loss/critic_ext_loss', critic_ext_loss, global_update)
            writer.add_scalar('loss/critic_int_loss', critic_int_loss, global_update)
            writer.add_scalar('loss/actor_loss', actor_loss, global_update)
            writer.add_scalar('loss/forward_loss', forward_loss, global_update)
            writer.add_scalar('loss/entropy', entropy, global_update)

            writer.add_scalar('reward/intrinsic_reward', np.sum(sample_batch.batch_int_reward) / num_worker,
                              global_update)
            writer.add_scalar('reward/mean_extrinsic_reward', 0 if not episode_rewards else np.mean(episode_rewards),
                              global_update)
            writer.add_scalar('reward/max_extrinsic_reward', 0 if not episode_rewards else np.max(episode_rewards),
                              global_update)

            writer.add_scalar('data/global_update', global_update, global_update)
            writer.add_scalar('data/episode_this_update', episode_this_update, global_update)
            writer.add_scalar('data/mean_steps_per_episode', episode_steps / episode_this_update, global_update)
            writer.add_scalar('data/mean_bomb_per_episode', count_bomb / episode_this_update,
                              global_update)

            writer.add_scalar('data/win_rate', episode_wins / episode_this_update, global_update)
            writer.add_scalar('data/tie_rate', episode_ties / episode_this_update, global_update)
            writer.add_scalar('data/loss_rate', episode_losses / episode_this_update, global_update)
            writer.add_scalar('data/ammo', episode_agent_ammo / episode_this_update, global_update)
            writer.add_scalar('data/blast_strength', episode_agent_blast_strength / episode_this_update, global_update)
            writer.add_scalar('data/can_kick', episode_agent_can_kick / episode_this_update, global_update)

            writer.add_scalar('value/intrinsic_value', np.mean(sample_batch.batch_int_values), global_update)
            writer.add_scalar('value/extrinsic_value', np.mean(sample_batch.batch_ext_values), global_update)
            writer.add_scalar('value/iv_explained',
                              explained_variance(sample_batch.batch_int_values[:, :-1].reshape([-1]),
                                                 sample_batch.int_target), global_update)
            writer.add_scalar('value/ev_explained',
                              explained_variance(sample_batch.batch_ext_values[:, :-1].reshape([-1]),
                                                 sample_batch.ext_target), global_update)

            print('Update: {} GlobalStep: {} #Episodes: {:3} AvgReward: {: .3f} WinRate: {:.3f}'.format(
                global_update,
                global_step,
                episode_this_update,
                np.mean(sample_batch.batch_ext_values),
                episode_wins / episode_this_update))

            episode_rewards.clear()
            episode_steps = 0
            count_bomb = 0
            episode_this_update = 0
            episode_wins = 0
            episode_ties = 0
            episode_losses = 0
            episode_agent_ammo = 0
            episode_agent_blast_strength = 0
            episode_agent_can_kick = 0

            torch.save(training_agent.model.state_dict(), model_path)
            torch.save(training_agent.rnd.predictor.state_dict(), predictor_path)
            torch.save(training_agent.rnd.target.state_dict(), target_path)


if __name__ == '__main__':
    main()
