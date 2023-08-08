from runners.AandB_runner import Runner_QMIX_AandB, Runner_STAS_AandB, Runner_COMA_AandB, Runner_SQDDPG_AandB
from envs.alice_and_bob.alice_and_bob import Alice_and_Bob
from utils.util import setup_seed
from arguments import *
import wandb

setup_seed(args.seed)
# print(args.model_save_path, args.tensorboard_save_path, args.log_save_path)

env = Alice_and_Bob()
n_agents = env.agent_num
n_action = env.n_action
obs_dim = env.obs_dim

if args.wandb:
    if args.method_name != 'STAS':
        wandb.init(project = 'RD-A&B', group = args.env_name + '-' + args.method_name)
    else:
        wandb.init(project = 'RD-A&B', group = args.env_name + '-' + args.method_name + '-' + args.reward_model_version)

if args.method_name == 'STAS':
    runner = Runner_STAS_AandB(args, args.env_name, env, device)
elif args.method_name == 'QMIX':
    runner = Runner_QMIX_AandB(args, args.env_name, env, device)
elif args.method_name == 'COMA':
    runner = Runner_COMA_AandB(args, args.env_name, env, device)
elif args.method_name == 'SQDDPG':
    runner = Runner_SQDDPG_AandB(args, args.env_name, env, device)
runner.run()