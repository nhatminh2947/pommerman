import numpy as np
import torch

import utils
from envs import make_vec_envs
from model import Policy, ActorCriticNetwork
import envs
from gym import spaces


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['reward'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))


if __name__ == '__main__':
    device = torch.device("cuda")
    checkpoint = torch.load('./trained_models/ppo/abc/PommeTeam-v0.pt')

    policy = checkpoint[0]
    policy.to(device)

    ob_rms = checkpoint[1]
    evaluate(policy, ob_rms, 'PommeTeam-v0', 2947, 1, device)
