"""
Implementation of PPO
ref: Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
ref: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
ref: https://github.com/openai/baselines/tree/master/baselines/ppo2
NOTICE:
    `Tensor2` means 2D-Tensor (num_samples, num_dims) 
"""

import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from collections import namedtuple
from torch.distributions import Categorical
import numpy as np
import math
from utils.norm import Normalizer
from utils.util import huber_loss, from_return_to_reward, generate_shuffle_indices, mask_select
import random
from arguments import *

Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward', 'info'))
EPS = 1e-10
# epsilon = 0.1

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_norm=True):
        super(ActorCritic, self).__init__()
        
        self.actor_fc1 = nn.Linear(num_inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, num_outputs)
        # self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)
        # self.actor = nn.Sequential(self.actor_fc1, self.actor_fc2, self.actor_fc3)
        # self.critic = nn.Sequential(self.critic_fc1, self.critic_fc2, self.critic_fc3)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_probs = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_probs, critic_value

    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_probs = self.actor_fc3(x)
        return action_probs

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def select_action(self, action_probs, greedy=False, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        dist = Categorical(logits=action_probs)
        action = dist.sample()

        if return_logproba:
            logproba = dist.log_prob(action)
        
        if greedy:
            action = torch.argmax(action_probs, dim=1)
        # action_std = torch.exp(action_logstd)
        # action = torch.normal(action_mean, action_std)
        # if return_logproba:
        #     logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_probs = self._forward_actor(states)
        dist = Categorical(logits=action_probs)
        logproba = dist.log_prob(actions)
        # logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba
    
class Memory(object):
    def __init__(self, max_step_per_round, n_agents):
        self.memory = []
        self.epi_length = []
        self.max_step_per_round = max_step_per_round
        self.n_agents = n_agents

    def push(self, *args):
        l = len(args[0])
        episode = [self.padding(arg, 0) if i!=2 else self.padding(arg, 4) for i, arg in enumerate(args)]
        self.memory.append(Transition(*episode))
        self.epi_length.append([l]*self.n_agents)

    def sample(self):
        return self.memory
        # return Transition(*zip(*self.memory))

    def padding(self, seq, pad):
        padded_seq = [np.full(seq[0].shape, pad) for i in range(self.max_step_per_round-len(seq))]
        return seq+padded_seq

    def __len__(self):
        return len(self.memory)

    def episode_length(self):
        return self.epi_length

class PPO(object):
    def __init__(self, num_inputs, num_actions, n_agents, num_epoch, max_step_per_round, num_episodes, 
                lr, clip, minibatch_size, gamma, lamda, 
                loss_coeff_value = 0.5, loss_coeff_entropy = 0.01, schedule_clip = 'linear', schedule_adam = 'linear', 
                value_norm = True, reward_norm = True, advantage_norm = True, lossvalue_norm = True, 
                layer_norm = True, return_decomposition = True, device='cpu'):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.n_agents = n_agents
        self.clip = clip
        self.clip_now = clip
        self.lr = lr
        self.num_epoch = num_epoch
        self.num_episodes = num_episodes
        self.max_step_per_round = max_step_per_round
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.lamda = lamda
        self.loss_coeff_entropy = loss_coeff_entropy
        self.loss_coeff_value = loss_coeff_value
        self.schedule_clip = schedule_clip
        self.schedule_adam = schedule_adam
        self.advantage_norm = advantage_norm
        self.value_norm = value_norm
        self.reward_norm = reward_norm
        self.lossvalue_norm = lossvalue_norm
        self.return_decomposition = return_decomposition
        self.huber_delta = 10
        self.max_grad_norm = 10
        self.device = device
        self.memory = Memory(self.max_step_per_round, self.n_agents)

        self.network = ActorCritic(num_inputs, num_actions, layer_norm=layer_norm).to(self.device)
        self.optimizer = opt.Adam(self.network.parameters(), lr=lr)
        self.running_state = ZFilter((n_agents, num_inputs), clip=5.0)
        self.value_normalizer = Normalizer(1).to(self.device)
    
    def state_norm(self, state):
        return self.running_state(state)
    
    def reset_memory(self):
        self.memory = Memory(self.max_step_per_round, self.n_agents)
        
    def select_action(self, state, epsilon, greedy=False):

        action_probs, value = self.network(state)
        # if avail_action != None:
        #     action_probs[torch.Tensor(avail_action) == 0] = -10000.
        action, logproba = self.network.select_action(action_probs, greedy)

        if epsilon is None or np.random.rand()>epsilon:
            action = action.data.cpu().numpy()
        else:
            action = np.array([np.random.randint(self.num_actions) for i in range(self.n_agents)])

        # action = action.data.cpu().numpy()
        logproba = logproba.data.cpu().numpy()
        value = value.data.cpu().numpy()
        return action, logproba, value

    def decompose_rewards_in_batch(self, batch, model, episode_length):
        model.eval()

        states = np.array(batch.state).transpose((0,2,1,3))
        states = torch.from_numpy(states).float()
        actions = np.squeeze(np.array(batch.action), axis=2).transpose((0,2,1))
        actions = torch.from_numpy(actions)
        pred_rewards = model(states.to(self.device), actions.to(self.device), episode_length)
        pred_rewards = pred_rewards.detach()
        return pred_rewards

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns. 

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip, self.clip)

        if self.value_norm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_loss = value_loss.mean()

        return value_loss

    def compute_returns(self, finalstates, pred_values, rewards, masks, episode_length):
        nextvalues = self.network._forward_critic(finalstates)
        # ipdb.set_trace()
        # pred_values = torch.cat((pred_values, nextvalues), dim=-1)
        pred_values = torch.cat((pred_values, torch.zeros(pred_values.shape[0], 1).to(self.device)), dim=-1)
        len_mask = (torch.arange(pred_values.shape[1])[None, :].to(self.device) == (episode_length-1)[:, None]).squeeze()
        pred_values[len_mask==1] = nextvalues.squeeze()
        returns = torch.zeros(pred_values.shape[0], self.max_step_per_round).to(self.device)

        gae = 0
        self.value_norm = False
        if self.value_norm:
            for i in reversed(range(self.max_step_per_round)):
                delta = rewards[:,i] + self.gamma * self.value_normalizer.denormalize(pred_values[:,i + 1]) \
                            * masks[:,i] - self.value_normalizer.denormalize(pred_values[:,i])
                gae = (delta + self.gamma * self.lamda * masks[:,i] * gae)*(i<episode_length)
                returns[:,i] = gae + self.value_normalizer.denormalize(pred_values[:,i])
        else:
            for i in reversed(range(self.max_step_per_round)):
                delta = rewards[:,i] + self.gamma * pred_values[:,i + 1] * masks[:,i] - pred_values[:,i]
                gae = (delta + self.gamma * self.lamda * masks[:,i] * gae)*(i<episode_length)
                returns[:,i] = gae + pred_values[:,i]
        return returns.detach()

    def update_parameters(self, batch_memory, reward_model, i_episode):
        batch = batch_memory.sample()
        episode_length = Tensor(np.array(batch_memory.episode_length())).to(self.device)
        batch = Transition(*zip(*batch))

        steps = self.max_step_per_round
        values = Tensor(np.array(batch.value)).squeeze().permute(0, 2, 1).reshape(-1, steps).to(self.device)
        masks = Tensor(np.array(batch.mask)).squeeze().permute(0, 2, 1).reshape(-1, steps).to(self.device)
        actions = mask_select(Tensor(np.array(batch.action)).squeeze().permute(0, 2, 1), episode_length, self.device)
        states = Tensor(np.array(batch.state)).squeeze().permute(0, 2, 1, 3).reshape(-1, steps, self.num_inputs).to(self.device)
        oldlogproba = mask_select(Tensor(np.array(batch.logproba)).squeeze().permute(0, 2, 1), episode_length, self.device)
        if self.return_decomposition:
            rewards = self.decompose_rewards_in_batch(batch, reward_model, episode_length[:,0]).reshape(-1, steps).to(self.device)
            rewards_statistics = (rewards.reshape(-1, self.n_agents, steps).detach().cpu().numpy(), 
                    np.squeeze(np.array(batch.reward), axis=2).transpose(0, 2, 1), np.array(batch.info).transpose(0, 2, 1))
        else:
            rewards = Tensor(np.array(batch.reward)).squeeze().permute(0, 2, 1).reshape(-1, steps).to(self.device)
            rewards_statistics = ()
        
        nextstates = Tensor(np.array(batch.next_state)).squeeze().permute(0, 2, 1, 3).reshape(-1, steps, self.num_inputs).to(self.device)
        len_mask = (torch.arange(steps)[None, :].to(self.device) == (episode_length-1).reshape(-1,1).squeeze()[:, None]).to(self.device).squeeze()
        nextstates = nextstates[len_mask==1]
        returns = self.compute_returns(nextstates, values, rewards, masks, episode_length.reshape(-1,1).squeeze())
        if self.value_norm:
            advantages = (returns - self.value_normalizer.denormalize(values))
        else:
            advantages = (returns - values)

        advantages = mask_select(advantages.reshape(-1, self.n_agents, steps), episode_length, self.device)
        values = mask_select(values.reshape(-1, self.n_agents, steps), episode_length, self.device)
        returns = mask_select(returns.reshape(-1, self.n_agents, steps), episode_length, self.device)
        states = mask_select(states.reshape(-1, self.n_agents, steps, self.num_inputs), episode_length, self.device)
        # print(rewards.shape, values.shape, masks.shape, actions.shape, states.shape, oldlogproba.shape)

        batch_size = actions.shape[0]
        if self.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        total_losses, loss_surrs, loss_values, loss_entropys = [], [], [], []
        for i_epoch in range(int(self.num_epoch * batch_size / self.minibatch_size)):
            # sample from current batch
            minibatch_ind = np.random.choice(batch_size, self.minibatch_size, replace=False)
            minibatch_states = states[minibatch_ind]
            minibatch_actions = actions[minibatch_ind]
            minibatch_oldlogproba = oldlogproba[minibatch_ind]
            minibatch_newlogproba = self.network.get_logproba(minibatch_states, minibatch_actions)
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            minibatch_values = values[minibatch_ind]
            minibatch_newvalues = self.network._forward_critic(minibatch_states)

            ratio =  torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
            surr1 = ratio * minibatch_advantages
            surr2 = ratio.clamp(1 - self.clip_now, 1 + self.clip_now) * minibatch_advantages
            loss_surr = - torch.mean(torch.min(surr1, surr2))

            # not sure the value loss should be clipped as well 
            # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
            # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
            # moreover, original paper does not mention clipped value 
            # if self.lossvalue_norm:
            #     minibatch_return_std = minibatch_returns.std()
            #     # print(minibatch_newvalues.squeeze().shape, minibatch_returns.shape, minibatch_return_6std[:,None].repeat(1, self.n_agents).shape)
            #     loss_value = torch.mean((minibatch_newvalues.squeeze() - minibatch_returns).pow(2)) / minibatch_return_std
            # else:
            #     loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))
            loss_value = self.cal_value_loss(minibatch_newvalues, minibatch_values, minibatch_returns)

            loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

            total_loss = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy
            total_losses.append(total_loss.item())
            loss_surrs.append(loss_surr.item())
            loss_values.append(loss_value.item())
            loss_entropys.append(loss_entropy.item())
            # print(total_loss, loss_surr, loss_value, loss_entropy)

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        if self.schedule_clip == 'linear':
            ep_ratio = 1 - (i_episode / self.num_episodes)
            self.clip_now = self.clip * ep_ratio

        if self.schedule_adam == 'linear':
            ep_ratio = 1 - (i_episode / self.num_episodes)
            lr_now = self.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in self.optimizer.param_groups:
                g['lr'] = lr_now

        return np.mean(total_losses), np.mean(loss_surrs), \
                np.mean(loss_values), np.mean(loss_entropys), rewards_statistics