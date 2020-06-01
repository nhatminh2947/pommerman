import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
import arguments
from algo.ppo import PPO
from algo.a2c_acktr import A2C_ACKTR
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage
from arguments import get_args
from model import ActorCriticNetwork
from gym import spaces


def main():
    writer = SummaryWriter()

    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.log_dir, device, False)

    policy = Policy(
        utils.OBS_SPACE_PER_AGENT.shape,
        envs.action_space,
        model=ActorCriticNetwork,
        base_kwargs={'recurrent': args.recurrent_policy})
    policy.to(device)

    if args.algo == 'a2c':
        agent = A2C_ACKTR(
            policy,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = PPO(
            policy,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = A2C_ACKTR(policy, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, utils.OBS_SPACE_PER_AGENT.shape,
                              envs.action_space, policy.recurrent_hidden_state_size)

    obs = envs.reset()  # tuple of 4 observation
    rollouts.obs[0].copy_(obs)  # store obs[0] for training
    rollouts.to(device)

    episode = {
        "reward": deque(maxlen=100),
        "step": deque(maxlen=100),
        "num_bombs": deque(maxlen=100)
    }

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = policy.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    for key in info['episode'].keys():
                        episode[key].append(info['episode'][key])

            # If done then clean the history of observations.
            masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.tensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = policy.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                policy,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode['reward']) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {}"
                  .format(j, total_num_steps, int(total_num_steps / (end - start))))
            print("\tdist_entropy, value_loss, action_loss: {:.5f}, {:.5f}, {:.5f}"
                  .format(dist_entropy, value_loss, action_loss))
            print("\tLast {} training episodes:".format(len(episode['reward'])))
            for key in episode.keys():
                print("\t\tmean/median {} {:.1f}/{:.1f}, min/max {} {:.1f}/{:.1f}"
                      .format(key, np.mean(episode[key]), np.median(episode[key]),
                              key, np.min(episode[key]), np.max(episode[key])))
            print()

            writer.add_scalar('Data/Updates', j, global_step=total_num_steps)
            writer.add_scalar('Data/Timesteps', total_num_steps, global_step=total_num_steps)
            writer.add_scalar('Data/AvgTimePerUpdate', ((end - start) / args.log_interval), global_step=total_num_steps)
            writer.add_scalar('Data/Entropy', dist_entropy, global_step=total_num_steps)
            writer.add_scalar('Data/ValueLoss', value_loss, global_step=total_num_steps)
            writer.add_scalar('Data/ActionLoss', action_loss, global_step=total_num_steps)
            for key in episode.keys():
                writer.add_scalar('Episode/mean_{}'.format(key), np.mean(episode[key]), global_step=total_num_steps)
                writer.add_scalar('Episode/max_{}'.format(key), np.max(episode[key]), global_step=total_num_steps)
                writer.add_scalar('Episode/min_{}'.format(key), np.min(episode[key]), global_step=total_num_steps)
            # if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
            #     ob_rms = utils.get_vec_normalize(envs).ob_rms
            #     evaluate(policy, ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)

            start = time.time()


if __name__ == "__main__":
    main()
