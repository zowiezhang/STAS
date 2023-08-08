import torch
import numpy as np
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils.replay_memory import ReplayBuffer
from models.policy.QMIX.qmix import QMIX
from utils.norm import Normalization
from utils.replay_memory import ReplayMemory_episode
from models.policy.ppo import PPO
from models.policy.COMA import COMA
from models.reward_model.mard.mard import STAS, STAS_ML
from utils.util import *
from utils.eval import eval_policy
from utils.plot import plot_reward
import wandb
import time
import os

class Runner_QMIX_AandB:
    def __init__(self, args, env_name, env, device):
        self.args = args
        self.env_name = env_name
        self.env = env
        self.device = device

        self.n_agents = env.agent_num
        self.n_action = env.n_action
        self.obs_dim = env.obs_dim
        self.state_dim = env.state_dim
        # print(self.n_agents, self.n_action, self.obs_dim, self.state_dim)

        # Create N agents
        self.agent_n = QMIX(self.args, self.n_agents, self.obs_dim, self.n_action, self.state_dim, device)
        self.replay_buffer = ReplayBuffer(self.args, self.n_agents, self.obs_dim, self.n_action, self.state_dim, device)

        self.epsilon = self.args.epsilon  # Initialize the epsilon
        # self.win_rates = []  # Record the win rates
        # self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)
        
        self.episode = 0

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        # while self.total_steps < self.args.max_train_steps:
        for i_episode in range(self.args.num_episodes):
            if self.episode // self.args.eval_freq > evaluate_num:
            # if i_episode % self.args.eval_freq == 0:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            for i in range(self.args.revision_factor):
                _, episode_reward, _ = self.run_episode_AandB(evaluate=False)  # Run an episode
                # self.total_steps += episode_steps
                if not self.args.wandb:
                    print('train episode reward: ', episode_reward, ', episode: {}'.format(i_episode))

                if self.replay_buffer.current_size >= self.args.batch_size:
                    loss = self.agent_n.train(self.replay_buffer, i_episode*self.args.revision_factor+i)  # Training
                    if not self.args.wandb:
                        print('QMIX loss: ', loss)
                    else:
                        wandb.log({'model loss': loss}, step = self.episode)

        self.evaluate_policy()
        # self.env.close()

    def evaluate_policy(self):
        win_times = 0
        evaluate_rewards = 0
        for _ in range(self.args.num_eval_runs):
            win_tag, episode_reward, _ = self.run_episode_AandB(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_rewards += episode_reward

        if self.args.wandb:
            wandb.log({'reach_treasure_rate': win_times / self.args.num_eval_runs, 'agent_reward': evaluate_rewards / self.args.num_eval_runs}, step=self.episode)
        else:
            print('evaluate policy, reach treasure rate: {:.4f}, agent reward: {:.4f}'.format(win_times / self.args.num_eval_runs, evaluate_rewards / self.args.num_eval_runs))

    def run_episode_AandB(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        init_state = self.env.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        last_onehot_a_n = np.zeros((self.n_agents, self.n_action))  # Last actions of N agents(one-hot)
        for episode_step in range(self.args.max_step_per_round):
            # obs_n = self.env.get_obs()  # obs_n.shape=(N, obs_dim)
            obs_n = self.env.get_obs()  
            # s = self.env.get_state()  # s.shape=(state_dim,)

            # s = self.env.get_obs()
            s = self.env.get_state()

            # avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            avail_a_n = np.ones([self.n_agents, self.n_action])
            epsilon = 0.1 if evaluate else self.epsilon
            # ipdb.set_trace()
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            last_onehot_a_n = np.eye(self.n_action)[a_n]  # Convert actions to one-hot vectors
            # r, done, info = self.env.step(a_n)  # Take a step

            next_obs_n, r, done, info = self.env.step(a_n)

            win_tag = True if all(done) else False
            episode_reward += np.sum(r)

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if all(done) and episode_step + 1 != self.args.max_step_per_round:
                    dw = True
                else:
                    dw = False

                # Store the transition
                if all(done) or episode_step == self.args.max_step_per_round - 1:
                    r = episode_reward
                else:
                    r = 0.0

                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min
                self.episode += 1

            if all(done):
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs() 
            s = self.env.get_state() 
            avail_a_n = np.ones([self.n_agents, self.n_action])

            # avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)

        return win_tag, episode_reward, episode_step + 1

class Runner_STAS_AandB:
    def __init__(self, args, env_name, env, device):
        self.args = args
        self.env_name = env_name
        self.env = env
        self.device = device
        self.n_agents = env.agent_num
        self.n_action = env.n_action
        self.obs_dim = env.obs_dim
        self.state_dim = env.state_dim

        self.agent = PPO(self.obs_dim, self.n_action, self.n_agents, args.num_epochs, args.max_step_per_round,
                    args.num_episodes, args.lr, args.clip, args.minibatch_size, args.gamma, args.lamda,
                    args.loss_coeff_value, args.loss_coeff_entropy, args.schedule_clip, args.schedule_adam, args.value_norm, args.reward_norm,
                    args.advantage_norm, args.lossvalue_norm, args.layer_norm, args.return_decomposition, device)
        self.eval_agent = PPO(self.obs_dim, self.n_action, self.n_agents, args.num_epochs, args.max_step_per_round,
                        args.num_episodes, args.lr, args.clip, args.minibatch_size, args.gamma, args.lamda,
                        args.loss_coeff_value, args.loss_coeff_entropy, args.schedule_clip, args.schedule_adam, args.value_norm, args.reward_norm,
                        args.advantage_norm, args.lossvalue_norm, args.layer_norm, args.return_decomposition, device)
        self.memory_e = ReplayMemory_episode(args.buffer_size, args.max_step_per_round, args.reward_norm)


        self.total_numsteps = 0
        self.running_episodes = 0
        self.epsilon = args.epsilon
        self.episode = 0

        time_length, state_emb = args.max_step_per_round, self.obs_dim
        if args.reward_model_version == 'v1':
            self.reward_model = STAS(input_dim=state_emb, n_actions = self.n_action, emb_dim = args.hidden_size, 
                                    n_heads=args.n_heads, n_layer=args.n_layers, seq_length=time_length, 
                                    n_agents=self.n_agents, sample_num = args.nums_sample_from_coalition, device = device, dropout=0.3)
        else:
            self.reward_model = STAS_ML(input_dim=state_emb, n_actions = self.n_action, emb_dim = args.hidden_size, 
                                    n_heads=args.n_heads, n_layer=args.n_layers, seq_length=time_length, 
                                    n_agents=self.n_agents, sample_num = args.nums_sample_from_coalition, device = device, dropout=0.3)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.reward_model = nn.DataParallel(self.reward_model)

        if torch.cuda.is_available() and args.cuda:
            self.reward_model.cuda()

        opt = torch.optim.Adam(lr=args.lr_reward, params=self.reward_model.parameters(), weight_decay=1e-5)
        loss_fn = nn.MSELoss(reduction='mean')
        # loss_fn = nn.SmoothL1Loss(beta=10)

        # Creates the train_step function for our model, loss function and optimizer
        self.train_step = make_train_step(self.reward_model, loss_fn, opt, self.n_agents, device, args.reg, args.alpha)

        # default `log_dir` is "runs" - we'll be more specific here
        self.writer = SummaryWriter(args.tensorboard_save_path)
        self.reward_record = []
        self.plot = True
    
    def run(self):
        evaluate_num = -1
        for i_episode in range(self.args.num_episodes):
            num_steps, ep_num = 0, 0
            reward_list, len_list, = [], []

            while num_steps < self.args.batch_size:
                win_tag, episode_reward, t, = self.run_episode_AandB(evaluate=False, i_episode=i_episode, ep_num=ep_num)

                ep_num += 1
                self.running_episodes += 1
                num_steps += (t + 1)
                self.total_numsteps += (t + 1)
                reward_list.append(episode_reward)
                len_list.append(t + 1)
            
            self.reward_record.append({
                    'episode': i_episode, 
                    'steps': self.total_numsteps, 
                    'meanepreward': np.mean(reward_list), 
                    'meaneplen': np.mean(len_list)})

            if len(self.memory_e)>self.args.reward_model_starts:
                policy_total_loss, loss_surr, loss_value, loss_entropy, r_statistics = self.agent.update_parameters(self.agent.memory, self.reward_model, i_episode)
                if i_episode % self.args.log_num_episode == 0:
                    if not self.args.wandb:
                        print('Finished episode: {} Reward: {:.4f} total_loss = {:.4f} = {:.4f} + {} * {:.4f} + {} * {:.4f}' \
                            .format(i_episode, reward_record[-1]['meanepreward'], policy_total_loss, loss_surr, self.args.loss_coeff_value, 
                            loss_value, self.args.loss_coeff_entropy, loss_entropy))
                        print('-----------------')
                    if self.args.return_decomposition:
                        decomposed_r, real_r, info = r_statistics
                        decomposed_r, real_r, info = rewards_process(decomposed_r, real_r, info)
                        if decomposed_r is not None:
                            self.plot = plot_reward(decomposed_r, real_r, info, i_episode, self.args.log_save_path, self.plot)
            ################################################
            # train the reward redistribution model
                
            if (i_episode+1) % self.args.reward_model_update_freq == 0 and (len(self.memory_e)>self.args.reward_model_starts):
            # if (len(self.memory_e)>40):
                epoch_train_total_reward_loss = []
                if self.args.reward_norm:
                    self.reward_model.module.reward_normalizer.update(self.memory_e.get_update_data())
                    self.memory_e.reset()
                for ii in range(self.args.updates_per_step):
                    states, actions, episode_return, episode_reward, episode_length = self.memory_e.sample_trajectory(n_trajectories=self.args.rewardbatch_size)
                    states = states.to(self.device)
                    actions = actions.to(self.device)
                    episode_return = episode_return.to(self.device)
                    episode_length = episode_length.to(self.device)
                    loss = self.train_step(states, actions, episode_return, episode_length)
                    epoch_train_total_reward_loss.append(loss)
                self.writer.add_scalar(self.args.exp_name + f'_reward_model_loss', np.mean(epoch_train_total_reward_loss), i_episode)
                if self.args.wandb:
                    wandb.log({'reward_model_loss': np.mean(epoch_train_total_reward_loss)}, step=self.episode)
                else:
                    print('Finished reward model training, episode: {} total_loss = {:.4f}' \
                        .format(i_episode, np.mean(epoch_train_total_reward_loss)))
                    print('-----------------')

            self.writer.add_scalar(self.args.exp_name + f'_episode_reward', self.reward_record[-1]['meanepreward'], self.running_episodes)
            self.episode += ep_num
            if self.episode // self.args.eval_freq > evaluate_num:
            # if i_episode % self.args.eval_freq == 0:
                self.evaluate_policy(i_episode)
                evaluate_num += 1
 
            self.agent.reset_memory()

        reward_record = pd.DataFrame(self.reward_record)
        reward_record.to_csv(os.path.join(self.args.log_save_path, 'policy_record.csv'))
        self.env.close()
        # torch.save({'reward_model': reward_model}, os.path.join(tr_log['model_save_path'], 'reward_model_final.ckpt'))
        time.sleep(5)
        # done_training.value = True

    def run_episode_AandB(self, evaluate, i_episode, ep_num):
        obs_n = self.env.reset()
        if self.args.state_norm:
            obs_n = self.agent.state_norm(obs_n)
        
        episode_reward = 0
        x_e, action_e, value_e, mask_e, x_next_e, logproba_e, reward_e, info_e = [], [], [], [], [], [], [], []
        for t in range(self.args.max_step_per_round):
            action_n, logproba_n, value_n = self.agent.select_action(torch.Tensor(obs_n).to(self.device), self.epsilon if not evaluate else 0, False)
            next_obs_n, reward_n, done_n, info_n = self.env.step(action_n)

            if self.args.state_norm:
                next_obs_n = self.agent.state_norm(next_obs_n)
            
            if not evaluate:
                if self.epsilon>0:
                    self.epsilon *= self.args.rate_decay

            episode_reward += np.sum(reward_n)
            x_e.append(np.array(obs_n))
            action_e.append(action_n.reshape(1,-1))
            value_e.append(value_n.reshape(1,-1))
            mask_e.append(np.array([[not done if t<self.args.max_step_per_round-1 else False for done in done_n]]))
            x_next_e.append(np.array(next_obs_n))
            logproba_e.append(logproba_n.reshape(1,-1))
            reward_e.append(np.array([reward_n]))
            info_e.append(np.array(info_n))
            
            if all(done_n) or t == self.args.max_step_per_round-1:
                win_tag = True if all(done_n) else False
                if not self.args.wandb:
                    mode = 'train' if not evaluate else 'eval'
                    print(mode, ' episode reward: ', episode_reward, ', episode/ep: {}/{}'.format(i_episode, ep_num))
                if not evaluate:
                    self.memory_e.push(x_e, action_e, mask_e, x_next_e, reward_e)
                    self.agent.memory.push(x_e, value_e, action_e, logproba_e, mask_e,
                            x_next_e, reward_e, info_e)
                break
            obs_n = next_obs_n
        
        return win_tag, episode_reward, t, # action_e, info_e

    def evaluate_policy(self, i_episode):
        win_times = 0
        evaluate_rewards = 0

        for _ in range(self.args.num_eval_runs):
            win_tag, episode_reward, t, = self.run_episode_AandB(evaluate=True, i_episode=i_episode, ep_num=_)
            if win_tag:
                win_times += 1
            evaluate_rewards += episode_reward
        
        if self.args.wandb:
            wandb.log({'reach_treasure_rate': win_times / self.args.num_eval_runs, 'agent_reward': evaluate_rewards / self.args.num_eval_runs}, step=self.episode)
        else:
            print('reach treasure rate: {:.4f}, agent reward: {:.4f}'.format(win_times / self.args.num_eval_runs, evaluate_rewards / self.args.num_eval_runs))
            
        # print(dev)
    
class Runner_COMA_AandB:
    def __init__(self, args, env_name, env, device):
        self.args = args
        self.env_name = env_name
        self.env = env
        self.device = device

        self.n_agents = env.agent_num
        self.n_action = env.n_action
        self.obs_dim = env.obs_dim
        self.state_dim = env.state_dim

        self.agent = COMA(self.n_agents, self.obs_dim, self.n_action, args.lr_critic, args.lr_actor, args.gamma, args.target_update_steps, device)
        self.episode = 0
    
    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        # while self.total_steps < self.args.max_train_steps:
        for i_episode in range(self.args.num_episodes):
            if self.episode // self.args.eval_freq > evaluate_num:
            # if i_episode % self.args.eval_freq == 0:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            for i in range(self.args.revision_factor):
                _, episode_reward, _ = self.run_episode_AandB(evaluate=False)  # Run an episode
                # self.total_steps += episode_steps
                if not self.args.wandb:
                    print('train episode reward: ', episode_reward, ', episode: {}'.format(i_episode))

                if self.episode % self.args.train_freq == 0:
                    actor_loss, critic_loss = self.agent.train()  # Training
                    if not self.args.wandb:
                        print('COMA actor loss: {:.4f}, critic loss: {:.4f}'.format(actor_loss, critic_loss))
                    else:
                        wandb.log({'COMA actor loss': actor_loss, 'COMA critic loss': critic_loss}, step = self.episode)

        self.evaluate_policy()
    
    def run_episode_AandB(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        obs_n = self.env.reset()
        for t in range(self.args.max_step_per_round):
            action_n = self.agent.get_actions(torch.Tensor(obs_n).to(self.device), evaluate)
            next_obs_n, reward_n, done_n, _ = self.env.step(action_n)

            episode_reward += np.sum(reward_n)
            win_tag = True if all(done_n) else False

            if not evaluate:
                if all(done_n) or t == self.args.max_step_per_round - 1:
                    r = [episode_reward for na in range(self.n_agents)]
                else:
                    r = [0.0 for na in range(self.n_agents)]
                if t == self.args.max_step_per_round - 1: done_n = [True for na in range(self.n_agents)]

                self.agent.memory.reward.append(r)
                for i in range(self.n_agents):
                    self.agent.memory.done[i].append(done_n[i])
                
                self.episode += 1

            obs_n = next_obs_n
        
            if all(done_n):
                break

        return win_tag, episode_reward, t
    
    def evaluate_policy(self):
        win_times = 0
        evaluate_rewards = 0
        for _ in range(self.args.num_eval_runs):
            win_tag, episode_reward, _ = self.run_episode_AandB(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_rewards += episode_reward

        if self.args.wandb:
            wandb.log({'reach_treasure_rate': win_times / self.args.num_eval_runs, 'agent_reward': evaluate_rewards / self.args.num_eval_runs}, step=self.episode)
        else:
            print('evaluate policy, reach treasure rate: {:.4f}, agent reward: {:.4f}'.format(win_times / self.args.num_eval_runs, evaluate_rewards / self.args.num_eval_runs))

from models.policy.sqddpg.util import *
from models.policy.sqddpg.inspector import *
from utils.replay_memory import TransReplayBuffer, EpisodeReplayBuffer
from torch import optim
from models.policy.sqddpg.sqddpg import SQDDPG
class Runner_SQDDPG_AandB(object):

    def __init__(self, args, env_name, env, device):
        self.args = args
        self.env_name = env_name
        self.env = env
        self.device = device
        self.online = args.online

        self.n_agents = env.agent_num
        self.n_action = env.n_action
        self.obs_dim = env.obs_dim

        self.args.agent_num = env.agent_num
        self.args.action_dim = env.n_action
        self.args.obs_size = env.obs_dim
        self.cuda_ = self.args.cuda and torch.cuda.is_available()

        inspector(self.args)
        if self.args.target:
            target_net = SQDDPG(self.args, self.n_agents, self.n_action, self.obs_dim).to(self.device)
            self.behaviour_net = SQDDPG(self.args, self.n_agents, self.n_action, self.obs_dim, target_net).to(self.device)
        else:
            self.behaviour_net = SQDDPG(self.args, self.n_agents, self.n_action, self.obs_dim).to(self.device)
        if self.args.replay:
            if self.online:
                self.replay_buffer = TransReplayBuffer(int(self.args.replay_buffer_size))
            else:
                self.replay_buffer = EpisodeReplayBuffer(int(self.args.replay_buffer_size))
        self.env = env
        self.action_optimizers = []
        for action_dict in self.behaviour_net.action_dicts:
            self.action_optimizers.append(optim.Adam(action_dict.parameters(), lr=args.policy_lrate))
        self.value_optimizers = []
        for value_dict in self.behaviour_net.value_dicts:
            self.value_optimizers.append(optim.Adam(value_dict.parameters(), lr=args.value_lrate))
        self.init_action = cuda_wrapper( torch.zeros(1, self.args.agent_num, self.args.action_dim), cuda=self.cuda_ )
        self.steps = 0
        self.episode = 0
        self.mean_reward = 0
        self.mean_success = 0
        self.entr = self.args.entr
        self.entr_inc = self.args.entr_inc

    def get_loss(self, batch):
        action_loss, value_loss, log_p_a = self.behaviour_net.get_loss(batch)
        return action_loss, value_loss, log_p_a

    def action_compute_grad(self, stat, loss, retain_graph):
        action_loss, log_p_a = loss
        if not self.args.continuous:
            if self.entr > 0:
                entropy = multinomial_entropy(log_p_a)
                action_loss -= self.entr * entropy
                stat['entropy'] = entropy.item()
        action_loss.backward(retain_graph=retain_graph)

    def value_compute_grad(self, value_loss, retain_graph):
        value_loss.backward(retain_graph=retain_graph)

    def grad_clip(self, params):
        for param in params:
            param.grad.data.clamp_(-1, 1)

    def action_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        return self.action_transition_process(stat, batch)

    def value_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        return self.value_transition_process(stat, batch)

    def action_transition_process(self, stat, trans):
        action_loss, value_loss, log_p_a = self.get_loss(trans)
        policy_grads = []
        for i in range(self.args.agent_num):
            retain_graph = False if i == self.args.agent_num-1 else True
            action_optimizer = self.action_optimizers[i]
            action_optimizer.zero_grad()
            self.action_compute_grad(stat, (action_loss[i], log_p_a[:, i, :]), retain_graph)
            grad = []
            for pp in action_optimizer.param_groups[0]['params']:
                grad.append(pp.grad.clone())
            policy_grads.append(grad)
        policy_grad_norms = []
        for action_optimizer, grad in zip(self.action_optimizers, policy_grads):
            param = action_optimizer.param_groups[0]['params']
            for i in range(len(param)):
                param[i].grad = grad[i]
            if self.args.grad_clip:
                self.grad_clip(param)
            policy_grad_norms.append(get_grad_norm(param))
            action_optimizer.step()
        stat['policy_grad_norm'] = np.array(policy_grad_norms).mean()
        stat['action_loss'] = action_loss.mean().item()
        return action_loss.mean().item()

    def value_transition_process(self, stat, trans):
        action_loss, value_loss, log_p_a = self.get_loss(trans)
        value_grads = []
        for i in range(self.args.agent_num):
            retain_graph = False if i == self.args.agent_num-1 else True
            value_optimizer = self.value_optimizers[i]
            value_optimizer.zero_grad()
            self.value_compute_grad(value_loss[i], retain_graph)
            grad = []
            for pp in value_optimizer.param_groups[0]['params']:
                grad.append(pp.grad.clone())
            value_grads.append(grad)
        value_grad_norms = []
        for value_optimizer, grad in zip(self.value_optimizers, value_grads):
            param = value_optimizer.param_groups[0]['params']
            for i in range(len(param)):
                param[i].grad = grad[i]
            if self.args.grad_clip:
                self.grad_clip(param)
            value_grad_norms.append(get_grad_norm(param))
            value_optimizer.step()
        stat['value_grad_norm'] = np.array(value_grad_norms).mean()
        stat['value_loss'] = value_loss.mean().item()

        return value_loss.mean().item()

    def run(self, ):
        stat = dict()
        evaluate_num = -1  # Record the number of evaluations
        # while self.total_steps < self.args.max_train_steps:
        for i_episode in range(self.args.num_episodes):
            if self.episode // self.args.eval_freq > evaluate_num:
            # if i_episode % self.args.eval_freq == 0:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            for i in range(self.args.revision_factor):
                _, episode_reward, _ = self.run_episode_AandB(stat, evaluate=False)  # Run an episode
                self.entr += self.entr_inc
                if not self.args.wandb:
                    print('train episode reward: ', episode_reward, ', episode: {}'.format(self.episode))

        self.evaluate_policy()

    def run_episode_AandB(self, stat, evaluate=False):
        info = {}
        win_tag = False
        episode_reward = 0
        state = self.env.reset()
        value_losses, action_losses = [], []
        if not evaluate and self.args.reward_record_type is 'episode_mean_step':
            self.mean_reward = 0
            self.mean_success = 0
        for t in range(self.args.max_step_per_round):
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_agents, self.obs_dim), self.cuda_)
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_agents, self.obs_dim), self.cuda_)
            action_out = self.behaviour_net.policy(state_, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info)
            _, actual = translate_action(self.args, action, self.env)
            # print(actual, [a.argmax() for a in actual])
            next_state, reward, done, debug = self.env.step([a.argmax() for a in actual])
            win_tag = True if all(done) else False
            episode_reward += np.sum(reward)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_step_per_round-1
            if not evaluate:
                if done_:
                    r = [episode_reward for na in range(self.n_agents)]
                else:
                    r = [0.0 for na in range(self.n_agents)]
                trans = self.behaviour_net.Transition(state,
                                        action.cpu().numpy(),
                                        np.array(r), #np.array(reward),
                                        next_state,
                                        done,
                                        done_
                                    )
                value_loss, action_loss = self.behaviour_net.transition_update(self, trans, stat)
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                success = debug['success'] if 'success' in debug else 0.0
                self.steps += 1
                if self.args.reward_record_type is 'mean_step':
                    self.mean_reward = self.mean_reward + 1/self.steps*(np.mean(reward) - self.mean_reward)
                    self.mean_success = self.mean_success + 1/self.steps*(success - self.mean_success)
                elif self.args.reward_record_type is 'episode_mean_step':
                    self.mean_reward = self.mean_reward + 1/(t+1)*(np.mean(reward) - self.mean_reward)
                    self.mean_success = self.mean_success + 1/(t+1)*(success - self.mean_success)
                else:
                    raise RuntimeError('Please enter a correct reward record type, e.g. mean_step or episode_mean_step.')
                stat['mean_reward'] = self.mean_reward
                stat['mean_success'] = self.mean_success
            if done_:
                break
            state = next_state
        if not evaluate:
            if not self.args.wandb:
                print('value loss: {:.4f}, action loss: {:.4f}'.format(np.mean(value_losses), np.mean(action_losses)))
            else:
                wandb.log({'SQDDPG value loss': np.mean(value_losses), 'SQDDPG action loss': np.mean(action_losses)}, step = self.episode)
            stat['turn'] = t+1
            self.episode += 1

        return win_tag, episode_reward, t

    def logging(self, stat):
        for tag, value in stat.items():
            if isinstance(value, np.ndarray):
                self.logger.image_summary(tag, value, self.episode)
            else:
                self.logger.scalar_summary(tag, value, self.episode)

    def print_info(self, stat):
        action_loss = stat.get('action_loss', 0)
        value_loss = stat.get('value_loss', 0)
        entropy = stat.get('entropy', 0)
        print ('Episode: {:4d}, Mean Reward: {:2.4f}, Action Loss: {:2.4f}, Value Loss is: {:2.4f}, Entropy: {:2.4f}\n'\
        .format(self.episode, stat['mean_reward'], action_loss+self.entr*entropy, value_loss, entropy))
    
    def evaluate_policy(self):
        win_times = 0
        evaluate_rewards = 0
        stat = dict()
        for _ in range(self.args.num_eval_runs):
            win_tag, episode_reward, _ = self.run_episode_AandB(stat, evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_rewards += episode_reward

        if self.args.wandb:
            wandb.log({'reach_treasure_rate': win_times / self.args.num_eval_runs, 'agent_reward': evaluate_rewards / self.args.num_eval_runs}, step=self.episode)
        else:
            print('evaluate policy, reach treasure rate: {:.4f}, agent reward: {:.4f}'.format(win_times / self.args.num_eval_runs, evaluate_rewards / self.args.num_eval_runs))