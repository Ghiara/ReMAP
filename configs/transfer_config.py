# default experiment settings
# all experiments should modify these settings only as needed
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.hopper_multi import HopperMulti
from sac_envs.walker_multi import WalkerMulti
from sac_envs.ant_multi_old import AntMulti

transfer_config = dict(

    experiment_name = 'cheetah_transfer',
    sim_time_steps = 20,
    max_path_len=100,
    batch_size = 20,
    policy_update_steps = 512,
    
    ### Define inference module to be reused
    
    # inference_path = '/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/toy1d-multi-task/2025_12_07_15_01_42_default_true_gmm_timesteps_48',
    

    inference_path = dict(
        name = '2026_05_24_16_23_33_default_dpmm_seed2_regular_loss_true_time_steps48_v1',
        path = '/root/bayes-tmp/bowang/Inference-reutilization-MRL/output/toy1d-multi-task/2026_05_24_16_23_33_default_dpmm_seed2_regular_loss_true_time_steps48_v1'
    ),


    ### Define the low-level controller and agent to reuse the inference mechanism

    complex_agent = dict(
        environment = WalkerMulti,
        experiments_repo = '/root/bayes-tmp/bowang/Inference-reutilization-MRL/output/low_level_policy/',
        experiment_name = 'walker_multi_new_config_v0',
        epoch = 1000,
    ),



    # complex_agent = dict(
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'walker_full_06_07',
    #     epoch = 2100,
    #     environment = WalkerMulti,
    # )
    # complex_agent = dict(
    #     environment = HopperMulti,
    #     experiments_repo = '/home/ubuntu/juan/Meta-RL/experiments_transfer_function/',
    #     experiment_name = 'hopper_full_sac0.2_reward1_randomchange',
    #     epoch = 1400,
    # )
    
)   
