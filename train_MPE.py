from runners.MPE_runner import Runner_QMIX_MPE, Runner_STAS_MPE, Runner_COMA_MPE, Runner_SQDDPG_MPE
from utils.util import setup_seed, make_env, n_actions
from arguments import *
import wandb
import numpy as np

setup_seed(args.seed)
# print(args.model_save_path, args.tensorboard_save_path, args.log_save_path)

env = make_env(args.scenario, discrete_action_input=False if args.method_name=='SQDDPG' else True)
n_agents = env.n
n_action = n_actions(env.action_space)[0]
obs_dim = env.observation_space[0].shape[0]

if args.wandb:
    if args.method_name != 'STAS':
        wandb.init(project = 'RD-MPE', group = args.env_name + '-' + args.scenario + '-' + args.method_name)
    else:
        wandb.init(project = 'RD-MPE', group = args.env_name + '-' + args.scenario + '-' + args.method_name)

if args.method_name == 'STAS':
    runner = Runner_STAS_MPE(args, args.env_name, env, n_agents, n_action, obs_dim, device)
elif args.method_name == 'QMIX':
    runner = Runner_QMIX_MPE(args, args.env_name, env, n_agents, n_action, obs_dim, device)
elif args.method_name == 'COMA':
    runner = Runner_COMA_MPE(args, args.env_name, env, n_agents, n_action, obs_dim, device)
elif args.method_name == 'SQDDPG':
    runner = Runner_SQDDPG_MPE(args, args.env_name, env, n_agents, n_action, obs_dim, device)
runner.run()