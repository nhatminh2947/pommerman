class SampleBatch:
    def __init__(self):
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
