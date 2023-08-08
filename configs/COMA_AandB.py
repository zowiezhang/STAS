import argparse
import torch
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Multi Agent Credit Assignment Under Delayed Return Decomposition Scneario')
parser.add_argument('--env_name', type=str, default='Alice_and_Bob', help='which env to test the algorithms.')
parser.add_argument('--method_name', type=str, default='COMA',
                    help='name of the method')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
parser.add_argument('--lr_actor', type=float, default=1e-4, help='learning rate for actor')
parser.add_argument('--lr_critic', type=float, default=1e-3, help='learning rate for critic')
parser.add_argument('--evaluate_cycle', type=int, default=5000, help='how often to evaluate the model')
parser.add_argument('--evaluate_epoch', type=int, default=32, help='number of the epoch to evaluate the agent')
parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
parser.add_argument('--reward_mode', type=str, default='accumulate', help="the reward mode: accumulate or extreme")
parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                help='batch size (default: 128)')
parser.add_argument('--target_update_steps', type=int, default=200, metavar='N')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--max_step_per_round', type=int, default=25, metavar='N',
                    help='max episode length (default: 1000)')                                          
parser.add_argument('--num_episodes', type=int, default=2000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--revision_factor', type=int, default=int(2048/40), metavar='N',
                    help='episode revision from ppo to q learning')
parser.add_argument('--wandb', default=True, action='store_false')             
parser.add_argument('--num_eval_runs', type=int, default=50, help='number of runs per evaluation (default: 5)')
parser.add_argument("--exp_name", type=str, help="name of the experiment")
parser.add_argument("--save_path", type=str, 
                    help="directory in which training state and model should be saved")
parser.add_argument('--cuda', default=True, action='store_true')
parser.add_argument('--eval_freq', type=int, default=1000)
parser.add_argument('--train_freq', type=int, default=20)

args = parser.parse_args()

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
