# Default RL2 configuration
# All RL2 experiments modify these settings as needed

default_config = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    n_eval_tasks=2,
    net_size=300,  # network width
    path_to_weights=None,
    env_params=dict(
        n_tasks=2,
        randomize_tasks=True,
    ),
    algo_params=dict(
        # Meta-training parameters
        meta_batch=16,
        num_iterations=500,
        num_initial_steps=2000,
        num_tasks_sample=5,
        num_steps_prior=400,
        num_steps_posterior=0,
        num_extra_rl_steps_posterior=400,
        num_train_steps_per_itr=2000,
        num_evals=2,
        num_steps_per_eval=600,
        
        # RL2-specific parameters
        inner_lr=1e-4,
        num_inner_steps=5,
        
        # Batch and network parameters
        batch_size=256,
        embedding_batch_size=64,
        embedding_mini_batch_size=64,
        max_path_length=200,
        
        # RL parameters
        discount=0.99,
        soft_target_tau=0.005,
        policy_lr=3e-4,
        qf_lr=3e-4,
        vf_lr=3e-4,
        context_lr=3e-4,
        
        # Other parameters
        reward_scale=5.0,
        sparse_rewards=False,
        kl_lambda=0.1,
        use_information_bottleneck=True,
        use_next_obs_in_context=False,
        update_post_train=1,
        num_exp_traj_eval=1,
        recurrent=True,  # RL2 uses recurrent policy
        dump_eval_paths=False,
    ),
    util_params=dict(
        base_log_dir='output',
        use_gpu=True,
        gpu_id=0,
        debug=False,
        docker=False,
    )
)
