import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pommerman.agents import BaseAgent
from pommerman.constants import Action
from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, RNDModel
from utils import global_grad_norm_


class StaticAgent(BaseAgent):
    """ Static agent"""

    def act(self, obs, action_space):
        return Action.Stop.value


class RNDAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False):
        print(input_size)
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)
        self.output_size = output_size
        self.input_size = input_size
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.rnd = RNDModel(input_size, output_size)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.rnd.predictor.parameters()),
                                    lr=learning_rate)
        self.rnd = self.rnd.to(self.device)

        self.model = self.model.to(self.device)

    def act(self, states):
        state = torch.from_numpy(states).to(self.device)
        state = state.float()
        policy, value_ext, value_int = self.model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()
        action = np.argmax(action_prob)
        # print('policy:', policy)
        # print('action_prob:', action_prob)
        # print('actions:', action)
        # print('value_ext:', value_ext)

        return action

    def get_action(self, states):
        state = torch.from_numpy(states).to(self.device)
        state = state.float()
        policy, value_ext, value_int = self.model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()
        actions = self.random_choice_prob_index(action_prob)

        # print('action_prob:', action_prob)
        # print('actions: ', actions)
        # print('value_ext: ', value_ext)
        # print('policy: ', policy)

        return actions, value_ext.data.cpu().numpy(), value_int.data.cpu().numpy(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, next_obs):
        next_obs = torch.from_numpy(next_obs).float().to(self.device)

        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy):
        s_batch = torch.from_numpy(s_batch).float().to(self.device)
        target_ext_batch = torch.from_numpy(target_ext_batch).float().to(self.device)
        target_int_batch = torch.from_numpy(target_int_batch).float().to(self.device)
        y_batch = torch.from_numpy(y_batch).long().to(self.device)
        adv_batch = torch.from_numpy(adv_batch).float().to(self.device)
        next_obs_batch = torch.from_numpy(next_obs_batch).float().to(self.device)

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        batch_forward_loss = []
        batch_actor_loss = []
        batch_critic_ext_loss = []
        batch_critic_int_loss = []
        batch_loss = []
        batch_entropy = []

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch[sample_idx])

                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                # ---------------------------------------------------------------------------------

                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_ext_loss

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy
                loss.backward()
                global_grad_norm_(list(self.model.parameters()) + list(self.rnd.predictor.parameters()))
                self.optimizer.step()

                batch_forward_loss.append(forward_loss.item())
                batch_actor_loss.append(actor_loss.item())
                batch_critic_ext_loss.append(critic_ext_loss.item())
                batch_critic_int_loss.append(critic_int_loss.item())
                batch_loss.append(loss.item())
                batch_entropy.append(entropy.item())

        loss = np.mean(batch_loss)
        critic_ext_loss = np.mean(batch_critic_ext_loss)
        critic_int_loss = np.mean(batch_critic_int_loss)
        actor_loss = np.mean(batch_actor_loss)
        forward_loss = np.mean(batch_forward_loss)
        entropy = np.mean(batch_entropy)

        return loss, critic_ext_loss, critic_int_loss, actor_loss, forward_loss, entropy
