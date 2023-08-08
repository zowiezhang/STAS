import argparse
import torch
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Multi Agent Credit Assignment Under Delayed Return Decomposition Scneario')
parser.add_argument('--env_name', type=str, default='MPE',
                    help='name of the environment type')
parser.add_argument('--scenario', required=True,
                    help='name of the environment to run')
parser.add_argument('--method_name', type=str, default='QMIX',
                    help='name of the method')
parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size (the number of episodes)")
parser.add_argument("--buffer_size", type=int, default=15000, help="The capacity of the replay buffer")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
parser.add_argument("--epsilon_decay_steps", type=float, default=30000, help="How many steps before the epsilon decays to the minimum")
parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS,if False, we will use Adam")
parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
parser.add_argument("--add_agent_id", type=bool, default=True, help="Whether to add agent id into the observation")
parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
parser.add_argument("--tau", type=int, default=0.01, help="If use soft update")
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--max_step_per_round', type=int, default=25, metavar='N',
                    help='max episode length (default: 1000)')                                          
parser.add_argument('--num_episodes', type=int, default=2000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--revision_factor', type=int, default=int(2048/25), metavar='N',
                    help='episode revision from ppo to q learning')
parser.add_argument('--wandb', default=True, action='store_false')             
parser.add_argument('--num_eval_runs', type=int, default=50, help='number of runs per evaluation (default: 5)')
parser.add_argument("--exp_name", type=str, help="name of the experiment")
parser.add_argument("--save_path", type=str, 
                    help="directory in which training state and model should be saved")
parser.add_argument('--cuda', default=True, action='store_true')
parser.add_argument('--eval_freq', type=int, default=500)

args = parser.parse_args()
args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

if args.exp_name is None:
    args.exp_name = 'Alice&Bob_STAS_hiddensize' \
                    + str(args.hidden_size) + '_' + str(args.seed)
else:
    args.exp_name = args.exp_name + '_' + str(args.seed)
print("=================Arguments==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

# torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
print('on device:', device)

if args.save_path is None:
    save_path = str(Path(os.path.abspath(__file__)).parents[0]) + '/results/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# create save folders
if 'model' not in os.listdir(save_path):
    os.mkdir(save_path+'model')
if 'tensorboard' not in os.listdir(save_path):
    os.mkdir(save_path+'tensorboard')
if 'log' not in os.listdir(save_path):
    os.mkdir(save_path+'log')
if args.exp_name not in os.listdir(save_path+'model/'):
    os.mkdir(save_path+'model/'+args.exp_name)
if args.exp_name not in os.listdir(save_path+'tensorboard/'):
    os.mkdir(save_path+'tensorboard/'+args.exp_name)
else:
    path = save_path+'tensorboard/'+args.exp_name
    for f in os.listdir(path):
        file_path = os.path.join(path,f)
        if os.path.isfile(file_path):
            os.remove(file_path)
if args.exp_name not in os.listdir(save_path+'log/'):
    os.mkdir(save_path+'log/'+args.exp_name)
else:
    path = save_path+'log/'+args.exp_name
    for f in os.listdir(path):
        file_path = os.path.join(path,f)
        if os.path.isfile(file_path):
            os.remove(file_path)

args.model_save_path = os.path.join(save_path, 'model', args.exp_name)
args.tensorboard_save_path = os.path.join(save_path, 'tensorboard', args.exp_name)
args.log_save_path = os.path.join(save_path, 'log', args.exp_name)
