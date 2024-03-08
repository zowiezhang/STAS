## Spatial Temporal Attention with Shapley value (STAS)
This is the code repository of paper [STAS: Spatial-Temporal Return Decomposition for Multi-agent Reinforcement Learning](https://arxiv.org/abs/2304.07520).

### Platform and Dependencies:

- python 3.7
- pytorch 1.7.1
- gym 0.10.9

### Install the improved MPE

Multi-Agent Particle Environment, named MPE, is a paricle world with a continuous observation and discrete action space, along with some basic simulated physics. Used in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf). The code is referred to https://github.com/openai/multiagent-particle-envs

Improved MPE extends the original MPE to a more complicated cooperative environment with multiple agents(from 3 to 15, etc.). Used in paper [PIC: Permutation Invariant Critic for Multi-Agent Deep Reinforcement Learning](https://arxiv.org/pdf/1911.00025.pdf). The code is referred to https://github.com/IouJenLiu/PIC

```shell
## Install the improved MPE
cd envs/multiagent-particle-envs
pip install -e .
```

### Training

To train the current version, use:

The argument 'STAS' means the current algorithm, there' s four to choose: STAS (ours), COMA, QMIX and SQDDPG. You can also change the reward_model_version to select the original STAS (v1) or STAS-ML (v2).

We use **wandb** by default as the default tool for experiment observation. If you prefer to use local logs, you can add ```--wandb``` at the end of the command to disable wandb.
```shell
## train in Alice and Bob
source train_AandB.sh STAS

## train in MPE
source train_MPE.sh STAS
```
Return decomposition model can be found in model/reward_model/mard/mard.py

Independent policy can be found in model/policy/ppo.py

All configs are set in configs/*