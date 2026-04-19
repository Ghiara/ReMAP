"""
Launcher for experiments with RL2
"""
import os
import pathlib
import numpy as np
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.rl2.networks import LSTMQFunction
from rlkit.torch.rl2.rl2_agent import RL2Agent
from rlkit.torch.rl2.rl2_sac import RL2SoftActorCritic
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu


# Keys consumed by RL2SoftActorCritic (not forwarded to MetaRLAlgorithm)
_RL2_KEYS = {
    'policy_lr', 'qf_lr', 'vf_lr',
    'soft_target_tau', 'policy_mean_reg_weight', 'policy_std_reg_weight',
    'inner_lr', 'num_inner_steps',
    'render_eval_paths', 'plotter',
}

# Keys that only PEARL uses (must be removed before passing to RL2)
_PEARL_ONLY_KEYS = {
    'context_lr', 'sparse_rewards', 'kl_lambda',
    'use_information_bottleneck', 'use_next_obs_in_context', 'recurrent',
}


def experiment(variant):
    """Run RL2 training experiment."""

    # Create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    print(f"Environment: {variant['env_name']}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Total tasks: {len(tasks)}, "
          f"Train: {variant['n_train_tasks']}, Eval: {variant['n_eval_tasks']}")

    net_size = variant['net_size']

    # LSTM Q-networks
    qf1 = LSTMQFunction(obs_dim=obs_dim, action_dim=action_dim,
                         hidden_size=net_size)
    qf2 = LSTMQFunction(obs_dim=obs_dim, action_dim=action_dim,
                         hidden_size=net_size)

    # MLP value function
    vf = FlattenMlp(hidden_sizes=[net_size, net_size, net_size],
                     input_size=obs_dim, output_size=1)

    # RL2 Agent (contains its own LSTMPolicy)
    agent = RL2Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=net_size,
        num_layers=1,
        policy_lr=variant['algo_params'].get('policy_lr', 3e-4),
    )

    # Strip PEARL-only keys from algo_params
    algo_params = {k: v for k, v in variant['algo_params'].items()
                   if k not in _PEARL_ONLY_KEYS}

    # Build algorithm
    algorithm = RL2SoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        **algo_params,
    )

    # Load pre-trained weights if provided
    if variant.get('path_to_weights') is not None:
        path = variant['path_to_weights']
        print(f"Loading pre-trained weights from {path}")
        agent.policy.load_state_dict(
            torch.load(os.path.join(path, 'policy.pth'), map_location='cpu'))
        qf1.load_state_dict(
            torch.load(os.path.join(path, 'qf1.pth'), map_location='cpu'))
        qf2.load_state_dict(
            torch.load(os.path.join(path, 'qf2.pth'), map_location='cpu'))
        vf.load_state_dict(
            torch.load(os.path.join(path, 'vf.pth'), map_location='cpu'))

    # GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'],
                     variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # Logging
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(
        variant['env_name'],
        variant=variant,
        exp_id=exp_id,
        base_log_dir=variant['util_params']['base_log_dir'],
    )

    if variant['algo_params'].get('dump_eval_paths', False):
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RL2 Training Configuration")
    print("=" * 80)
    print(json.dumps(variant, indent=2, default=str))
    print("=" * 80)

    # Run training
    algorithm.train()


def deep_update_dict(fr, to):
    """Update dict of dicts with new values."""
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to
