import glob
import os

import torch
import torch.nn as nn
from pommerman import constants
import numpy as np
from envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def featurize(obs):
    id = 0
    features = np.zeros(shape=(16, 11, 11))
    # print(obs)
    for item in constants.Item:
        if item in [constants.Item.Bomb,
                    constants.Item.Flames,
                    constants.Item.Agent0,
                    constants.Item.Agent1,
                    constants.Item.Agent2,
                    constants.Item.Agent3,
                    constants.Item.AgentDummy]:
            continue
        # print("item:", item)
        # print("board:", obs["board"])

        features[id, :, :][obs["board"] == item.value] = 1
        id += 1

    for feature in ["flame_life", "bomb_life", "bomb_blast_strength"]:
        features[id, :, :] = obs[feature]
        id += 1

    features[id, :, :][obs["position"]] = 1
    id += 1

    features[id, :, :][obs["board"] == obs["teammate"].value] = 1
    id += 1

    for enemy in obs["enemies"]:
        features[id, :, :][obs["board"] == enemy.value] = 1
    id += 1

    features[id, :, :] = np.full(shape=(11, 11), fill_value=obs["ammo"])
    id += 1

    features[id, :, :] = np.full(shape=(11, 11), fill_value=obs["blast_strength"])
    id += 1

    features[id, :, :] = np.full(shape=(11, 11), fill_value=(1 if obs["can_kick"] else 0))
    id += 1

    return features
