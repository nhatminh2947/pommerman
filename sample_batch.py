import numpy as np
from pommerman import constants
from constants import N_CHANNELS
from utils import RunningMeanStd, RewardForwardFilter, make_train_data


class SampleBatch:
    def __init__(self, ext_gamma=0.999, int_gamma=0.99, num_steps=128, num_workers=32, int_coef=1., ext_coef=2.):
        self.batch_states = []
        self.batch_reward = []
        self.batch_done = []
        self.batch_next_state = []
        self.batch_action = []
        self.batch_int_reward = []
        self.batch_next_obs = []
        self.batch_ext_values = []
        self.batch_int_values = []
        self.batch_policy = []
        self.batch_adv = []
        self.reward_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(int_gamma)
        self.num_steps = num_steps
        self.ext_gamma = ext_gamma
        self.int_gamma = int_gamma
        self.num_workers = num_workers
        self.int_coef = int_coef
        self.ext_coef = ext_coef

    def reset(self):
        self.batch_states = []
        self.batch_reward = []
        self.batch_done = []
        self.batch_next_state = []
        self.batch_action = []
        self.batch_int_reward = []
        self.batch_next_obs = []
        self.batch_ext_values = []
        self.batch_int_values = []
        self.batch_policy = []
        self.batch_adv = []

    def add(self, next_obs, intrinsic_reward, states, rewards, dones, actions, value_ext, value_int, policy):
        self.batch_next_obs.append(next_obs)
        self.batch_int_reward.append(intrinsic_reward)
        self.batch_states.append(states)
        self.batch_reward.append(rewards)
        self.batch_done.append(dones)
        self.batch_action.append(actions)
        self.batch_ext_values.append(value_ext)
        self.batch_int_values.append(value_int)
        self.batch_policy.append(policy)

    def add_last_next_value(self, value_ext, value_int):
        self.batch_ext_values.append(value_ext)
        self.batch_int_values.append(value_int)

    def preprocess(self):
        # self.batch_states = np.stack(self.batch_states).transpose([1, 0, 2, 3, 4]).reshape(
        #     [-1, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE])
        self.batch_reward = np.stack(self.batch_reward).transpose()
        self.batch_action = np.stack(self.batch_action).transpose().reshape([-1])
        self.batch_done = np.stack(self.batch_done).transpose()
        # self.batch_next_obs = np.stack(self.batch_next_obs).transpose([1, 0, 2, 3, 4]).reshape(
        #     [-1, N_CHANNELS, constants.BOARD_SIZE, constants.BOARD_SIZE])
        self.batch_ext_values = np.squeeze(self.batch_ext_values, axis=-1).transpose()
        self.batch_int_values = np.squeeze(self.batch_int_values, axis=-1).transpose()

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        self.batch_int_reward = np.stack(self.batch_int_reward).transpose()
        total_reward_per_env = np.array([self.discounted_reward.update(reward_per_step) for reward_per_step in
                                         self.batch_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        self.reward_rms.update_from_moments(mean, std ** 2, count)
        # normalize intrinsic reward
        self.batch_int_reward /= np.sqrt(self.reward_rms.var)

        # Step 3. make target and advantage
        # extrinsic reward calculate
        self.ext_target, ext_adv = make_train_data(self.batch_reward,
                                              self.batch_done,
                                              self.batch_ext_values,
                                              self.ext_gamma,
                                              self.num_steps,
                                              self.num_workers)

        # intrinsic reward calculate
        # None Episodic
        self.int_target, int_adv = make_train_data(self.batch_int_reward,
                                              np.zeros_like(self.batch_int_reward),
                                              self.batch_int_values,
                                              self.int_gamma,
                                              self.num_steps,
                                              self.num_workers)

        # add ext adv and int adv
        self.batch_adv = int_adv * self.int_coef + ext_adv * self.ext_coef
