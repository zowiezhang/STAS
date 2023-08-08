import argparse
import torch
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Multi Agent Credit Assignment Under Delayed Return Decomposition Scneario')
parser.add_argument('--env_name', type=str, default='MPE', help='which env to test the algorithms.')
parser.add_argument('--method_name', type=str, default='SQDDPG',
                    help='name of the method')
parser.add_argument('--scenario', required=True,
                    help='name of the environment to run')
parser.add_argument('--hid_size', type=int, default=128)
parser.add_argument('--sample_size', type=int, default=2)
parser.add_argument('--continuous', type=bool, default=False)
parser.add_argument('--init_std', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
parser.add_argument('--policy_lrate', type=float, default=1e-4, help='learning rate for actor')
parser.add_argument('--value_lrate', type=float, default=1e-4, help='learning rate for critic')
parser.add_argument('--normalize_advantages', type=bool, default=False)
parser.add_argument('--entr', type=float, default=1e-3)
parser.add_argument('--entr_inc', type=float, default=0.0)
parser.add_argument('--q_func', type=bool, default=True)
parser.add_argument('--replay', type=bool, default=True)
parser.add_argument('--replay_buffer_size', type=int, default=1.5e4)
parser.add_argument('--replay_warmup', type=int, default=0)
parser.add_argument('--grad_clip', type=bool, default=True)
parser.add_argument('--save_model_freq', type=int, default=10)
parser.add_argument('--target', type=bool, default=True)
parser.add_argument('--target_lr', type=float, default=1e-1)
parser.add_argument('--behaviour_update_freq', type=int, default=100)
parser.add_argument('--critic_update_times', type=int, default=10)
parser.add_argument('--target_update_freq', type=int, default=200)
parser.add_argument('--gumbel_softmax', type=bool, default=True)
parser.add_argument('--epsilon_softmax', type=bool, default=False)
parser.add_argument('--online', type=bool, default=True)
parser.add_argument("--reward_record_type", type=str, default="episode_mean_step")
parser.add_argument('--shared_parameters', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                help='batch size (default: 128)')
parser.add_argument('--target_update_steps', type=int, default=200, metavar='N')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--max_step_per_round', type=int, default=200, metavar='N',
                    help='max episode length (default: 1000)')                                          
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--revision_factor', type=int, default=int(2048/25), metavar='N',
                    help='episode revision from ppo to q learning')
parser.add_argument('--wandb', default=True, action='store_false')             
parser.add_argument('--num_eval_runs', type=int, default=10, help='number of runs per evaluation (default: 5)')
parser.add_argument("--exp_name", type=str, help="name of the experiment")
parser.add_argument("--save_path", type=str, 
                    help="directory in which training state and model should be saved")
parser.add_argument('--cuda', default=True, action='store_true')
parser.add_argument('--eval_freq', type=int, default=100)

args = parser.parse_args()

if args.exp_name is None:
    args.exp_name = 'Alice&Bob_SQDDPG_hiddensize' \
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
