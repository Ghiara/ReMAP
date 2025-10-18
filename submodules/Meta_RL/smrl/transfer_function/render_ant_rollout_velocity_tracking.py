"""
Render Ant agent (velocity tracking) using trained weights and save rollout video.
"""

import os
import torch
import imageio
import numpy as np
import cv2

from sac_envs.ant_multi_old import AntMulti
from model import PolicyNetwork as TransferFunction

# ========== CONFIG ==========
EXPERIMENTS_REPO = "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/low_level_policy"
EXPERIMENT_NAME  = "ant_multi_old_architecture_velocity_left"   # ⚠️ 改成你训练 velocity 的实验名
EPOCH            = 500
OUT_VIDEO        = f"ant_velocity_rollout_epoch{EPOCH}.mp4"
MAX_PATH_LENGTH  = 500
FPS              = 30
# ============================


def get_policy_and_env():
    """加载 Ant 环境和对应权重"""
    config_path = os.path.join(EXPERIMENTS_REPO, EXPERIMENT_NAME, "config.json")
    env_config = {}
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            env_config = json.load(f)

    # 创建 Ant 环境
    env = AntMulti(env_config, render_mode="rgb_array")

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]

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


def rollout_and_render(env, policy, max_path_length=500, out_video="ant_velocity.mp4", fps=30):
    """在 Mujoco 环境中 rollout 并保存为视频（velocity tracking）"""
    obs, _ = env.reset()

    # ---- 设置目标速度 ----
    target_velocity = np.random.uniform(-3.0, -1.0)    # 可以改成 (-2.0, -0.5) 看向左走
    new_task = np.array([target_velocity])           # ⚠️ task 长度必须与训练时一致
    if hasattr(env, "update_task"):
        env.update_task(new_task)
    print("New target velocity set to:", new_task)

    frames = []
    prev_xpos = env.get_body_com("torso")[0]

    for t in range(max_path_length):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        # 获取任务向量
        if hasattr(env, "task"):
            task = env.task
        else:
            task = np.zeros((1, 1))
        task_tensor = torch.as_tensor(task, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = policy.get_action(obs_tensor, task_tensor)
            action = action.cpu().numpy().squeeze(0)

        obs, reward, done, _, _ = env.step(action)

        # ---- 计算当前速度 ----
        xpos = env.get_body_com("torso")[0]
        forward_vel = (xpos - prev_xpos) / env.dt
        prev_xpos = xpos

        frame = env.sim.render(width=640, height=480, camera_name="track")
        frame = np.flipud(frame)
        frame = np.ascontiguousarray(frame)

        # ---- 在画面上叠加信息 ----
        y = 25
        cv2.putText(frame, f"Step: {t}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += 25
        cv2.putText(frame, f"Reward: {reward:.3f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 25
        cv2.putText(frame, f"Target velocity: {task[0]:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        y += 25
        cv2.putText(frame, f"Current velocity: {forward_vel:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)

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
