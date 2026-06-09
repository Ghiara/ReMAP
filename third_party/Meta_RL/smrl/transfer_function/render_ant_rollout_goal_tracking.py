# """
# Render Ant agent using trained weights and save rollout video.
# """

# import os
# import torch
# import imageio
# import numpy as np

# from sac_envs.ant_multi_old import AntMulti
# from model import PolicyNetwork as TransferFunction

# # ========== CONFIG ==========
# EXPERIMENTS_REPO = "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/low_level_policy"
# EXPERIMENT_NAME  = "ant_multi_old_architecture_only_goal_left_with_termination_goal_tracking"
# EPOCH            = 400
# OUT_VIDEO        = f"ant_rollout_epoch{EPOCH}.mp4"
# MAX_PATH_LENGTH  = 500
# FPS              = 30
# # ============================


# def get_policy_and_env():
#     """加载 Ant 环境和对应权重"""

#     # 配置文件路径（如果缺失就用空dict）
#     config_path = os.path.join(EXPERIMENTS_REPO, EXPERIMENT_NAME, "config.json")
#     env_config = {}
#     if os.path.exists(config_path):
#         import json
#         with open(config_path, "r") as f:
#             env_config = json.load(f)

#     # 创建 Ant 环境
#     env = AntMulti(env_config, render_mode="rgb_array")

#     # 观测维度和动作维度
#     n_states = env.observation_space.shape[0]
#     n_actions = env.action_space.shape[0]
#     action_bounds = [env.action_space.low[0], env.action_space.high[0]]

#     # 加载 policy 权重
#     pretrained = os.path.join(
#         EXPERIMENTS_REPO,
#         EXPERIMENT_NAME,
#         "models",
#         "policy_model",
#         f"epoch_{EPOCH}.pth",
#     )
#     print("Load model", pretrained)
#     policy = TransferFunction(
#         n_states=n_states,
#         n_actions=n_actions,
#         action_bounds=action_bounds,
#         pretrained=pretrained,
#     )
#     policy.eval()

#     return env, policy


# def rollout_and_render(env, policy, max_path_length=500, out_video="ant.mp4", fps=30):
#     obs, _ = env.reset()
#     frames = []

#     for t in range(max_path_length):
#         obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

#         # 取任务向量
#         if hasattr(env, "task"):
#             task = env.task
#         else:
#             task = np.zeros((1, 1))   # fallback: 单任务占位
#         task_tensor = torch.as_tensor(task, dtype=torch.float32).unsqueeze(0)

#         with torch.no_grad():
#             action = policy(obs_tensor, task_tensor).cpu().numpy().squeeze(0)

#         obs, reward, done, _, _ = env.step(action)

#         frame = env.render(mode="rgb_array")
#         if frame is not None:
#             frames.append(frame)

#         if done:
#             break

#     imageio.mimsave(out_video, frames, fps=fps)
#     print(f"[OK] 视频已保存到 {out_video}")



# def main():
#     env, policy = get_policy_and_env()
#     rollout_and_render(env, policy, MAX_PATH_LENGTH, OUT_VIDEO, FPS)


# if __name__ == "__main__":
#     main()



"""
Render Ant agent using trained weights and save rollout video.
"""

import os
import torch
import imageio
import numpy as np

from third_party.SAC.sac_envs.ant_multi import AntMulti
from third_party.SAC.model import PolicyNetwork as TransferFunction

# ========== CONFIG ==========
EXPERIMENTS_REPO = "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/low_level_policy"
EXPERIMENT_NAME  = "ant_multi_old_architecture_only_real_goal_left"
EPOCH            = 500
OUT_VIDEO        = f"ant_rollout_epoch{EPOCH}.mp4"
MAX_PATH_LENGTH  = 500
FPS              = 30
# ============================


def get_policy_and_env():
    """加载 Ant 环境和对应权重"""

    # 配置文件路径（如果缺失就用空dict）
    config_path = os.path.join(EXPERIMENTS_REPO, EXPERIMENT_NAME, "config.json")
    env_config = {}
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            env_config = json.load(f)

    # 创建 Ant 环境
    env = AntMulti(env_config, render_mode="rgb_array")

    # 观测维度和动作维度
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]

    # 加载 policy 权重
    pretrained = os.path.join(
        EXPERIMENTS_REPO,
        EXPERIMENT_NAME,
        "models",
        "policy_model",
        f"epoch_{EPOCH}.pth",
    )
    print("Load model", pretrained)
    policy = TransferFunction(
        n_states=n_states,
        n_actions=n_actions,
        action_bounds=action_bounds,
        pretrained=pretrained,
    )
    policy.eval()

    return env, policy


def rollout_and_render(env, policy, max_path_length=500, out_video="ant.mp4", fps=30):
    """在 Mujoco 环境中 rollout 并保存为视频"""

    obs, _ = env.reset()
    # 手动设置更远的目标
    goal_distance = np.random.uniform(-10.0, -6.0)   # 6 到 10 米
    new_task = np.array([goal_distance])       # 例如 (x=0, y=目标)
    if hasattr(env, "update_task"):
        env.update_task(new_task)
    print("New goal set to:", new_task)

    frames = []

    for t in range(max_path_length):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        # 取任务向量
        if hasattr(env, "task"):
            task = env.task
        else:
            task = np.zeros((1, 1))   # fallback: 单任务占位
        task_tensor = torch.as_tensor(task, dtype=torch.float32).unsqueeze(0)

        # 用 get_action 得到实际动作
        with torch.no_grad():
            action = policy.get_action(obs_tensor, task_tensor)
            action = action.cpu().numpy().squeeze(0)

        obs, reward, done, _, _ = env.step(action)

        frame = env.sim.render(width=640, height=480, camera_name="track")
        frame = np.flipud(frame) 
        frame = np.ascontiguousarray(frame)

        # ----- 在画面上叠加信息 -----
        import cv2
        info_y = 20
        cv2.putText(frame, f"Step: {t}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        info_y += 25
        cv2.putText(frame, f"Reward: {reward:.3f}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        info_y += 25
        cv2.putText(frame, f"Task: {task}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        if frame is not None:
            frames.append(frame)

        if done:
            break

    # 保存视频
    if frames:
        imageio.mimsave(out_video, frames, fps=fps)
        print(f"[OK] 视频已保存到 {out_video}")
    else:
        print("[WARN] 没有捕获到任何渲染帧，请检查 env.render() 是否正常工作")


def main():
    env, policy = get_policy_and_env()
    rollout_and_render(env, policy, MAX_PATH_LENGTH, OUT_VIDEO, FPS)


if __name__ == "__main__":
    main()

    #print("obs_dim:", obs_tensor.shape[1], "task_dim:", task_tensor.shape[1])

