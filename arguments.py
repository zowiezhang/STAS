import argparse
import torch
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Multi Agent Credit Assignment Under Delayed Return Decomposition Scneario')
parser.add_argument('--env_name', type=str, default='Alice_and_Bob',
                    help='name of the environment type')
parser.add_argument('--method_name', type=str, default='STAS',
                    help='name of the method')
parser.add_argument('--gamma', type=float, default=1, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.95, metavar='G',
                    help='GAE factor')                 
parser.add_argument('--num_epochs', type=int, default=10, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--max_step_per_round', type=int, default=25, metavar='N',
                    help='max episode length (default: 1000)')                    
parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--minibatch_size', type=int, default=256, metavar='N',
                    help='minibatch size (default: 128)')
parser.add_argument('--rewardbatch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 128)')                        
parser.add_argument('--num_episodes', type=int, default=2000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=50, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--n_layers', type=int, default=3, metavar='N',
                    help='number of layers of reward module')
parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                    help='number of heads of reward module')                    
parser.add_argument('--buffer_size', type=int, default=20000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--clip', type=float, default=0.2)
parser.add_argument('--epsilon', type=float, default=0.95,)
parser.add_argument('--rate_decay', type=float, default=0.995,)
parser.add_argument('--lr', type=float, default=1e-4,
                    help='(default: 1e-4)')
parser.add_argument('--lr_reward', type=float, default=1e-4,
                    help='learning rate for reward model')
parser.add_argument('--loss_coeff_entropy', type=float, default=0.01)
parser.add_argument('--loss_coeff_value', type=float, default=0.5)
parser.add_argument("--schedule_clip", type=str, default='linear', help="linear decay clip rate")
parser.add_argument("--schedule_adam", type=str, default='linear', help="linear decay learning rate")
parser.add_argument('--advantage_norm', default=True, action='store_false')
parser.add_argument('--value_norm', default=True, action='store_false')
parser.add_argument('--reward_norm', default=False, action='store_true')
parser.add_argument('--wandb', default=True, action='store_false')
parser.add_argument('--state_norm', default=False, action='store_true')
parser.add_argument('--lossvalue_norm', default=False, action='store_true')
parser.add_argument('--layer_norm', default=True, action='store_false')
parser.add_argument('--return_decomposition', default=True, action='store_false')
parser.add_argument('--reg', default=False, action='store_true')
parser.add_argument('--alpha', type=float, default=2.0)
parser.add_argument('--nums_sample_from_coalition', type=int, default=3, help='number of MC estimate shapley value')                                     
parser.add_argument('--num_eval_runs', type=int, default=50, help='number of runs per evaluation (default: 5)')
parser.add_argument('--log_num_episode', type=int, default=20, metavar='N')
parser.add_argument("--exp_name", type=str, help="name of the experiment")
parser.add_argument("--save_path", type=str, 
                    help="directory in which training state and model should be saved")
parser.add_argument('--cuda', default=True, action='store_true')
parser.add_argument('--eval_freq', type=int, default=100)
parser.add_argument('--reward_model_update_freq', type=int, default=10)
parser.add_argument('--reward_model_starts', type=int, default=10000)
parser.add_argument("--reward_model_version", type=str, default='v2')

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
