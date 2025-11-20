from mrl_analysis.video.video_creator_replay import VideoCreatorReplay, load_replay_data
from sac_envs.ant_multi_old import AntMulti  # 或者其他环境

# 1. 加载数据
trajectories, env_name, config, env_config= load_replay_data(
    "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/submodules/Meta_RL/smrl/transfer_function/evaluation/experiments_thesis/transfer_ant_test/toy1d_MaxAction_1_2025-02-05_14-55-41/replay_data.pkl"
)


import numpy as np

# 假设你已经有 trajectories 变量
traj0 = trajectories[0]

print("\n=== DEBUG: Trajectory structure ===")
print("Keys:", traj0.keys())

print("\n=== DEBUG: Actions shape and sample ===")
print("Actions shape:", np.array(traj0["actions"]).shape)
print("First 5 actions:\n", np.array(traj0["actions"][:5]))
print("Mean action magnitude:", np.mean(np.abs(traj0["actions"])))

print("\n=== DEBUG: Observations shape ===")
print(np.array(traj0["observations"]).shape)


# 2. 重新构造环境（如果需要）
env = AntMulti(env_config)  # 可从 config 重建环境
env.reset()

# 3. 生成视频
video_creator = VideoCreatorReplay(fps=30, width=800, height=600)
video_creator.create_video_from_trajectories(
    env=env,
    trajectories=trajectories,
    save_as="/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/submodules/Meta_RL/smrl/transfer_function/evaluation/experiments_thesis/transfer_ant_test/toy1d_MaxAction_1_2025-02-05_14-55-41/replay/ant_replay.mp4",
    show_info=True,
    overlay_latent=True,
    overlay_goal=True,
    overlay_reward=True,
)
