"""
Render Cheetah agent (velocity tracking) using multi-task trained weights
and save rollout video for checking pure low-level velocity tracking ability.

This is the multi-task version of your Ant single-task evaluator.
"""

import os
import torch
import imageio
import numpy as np
import cv2

# ========== TODO: 改成你自己项目路径 ==========
from third_party.SAC.sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from third_party.SAC.model import PolicyNetwork as LowLevelPolicy
# =============================================

# ========== CONFIG ==========
EXPERIMENTS_REPO = "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/low_level_policy"
EXPERIMENT_NAME  = "new_cheetah_training/half_cheetah_initial_random"     # ⚠️ 改成你的 cheetah multi-task 训练目录名
EPOCH            =1100                          # 想渲染第几个 epoch 的模型
OUT_VIDEO        = f"cheetah_multi_velocity_epoch{EPOCH}.mp4"
MAX_PATH_LENGTH  = 500
FPS              = 30
# ============================


def load_env_and_policy():
    """加载 Cheetah multi-task 环境与低层策略模型"""
    
    # ---- 读取 config（可选但推荐）----
    config_path = os.path.join(EXPERIMENTS_REPO, EXPERIMENT_NAME, "config.json")
    env_config = {}
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            env_config = json.load(f)
    print("Loaded config from:", config_path)

    # ---- 创建环境 ----
    env = HalfCheetahMixtureEnv(env_config, render_mode="rgb_array")

    # ---- 获取状态/动作维度 ----
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]

    # ---- 加载 policy ----
    pretrained = os.path.join(
        EXPERIMENTS_REPO,
        EXPERIMENT_NAME,
        "models",
        "policy_model",
        f"epoch_{EPOCH}.pth",
    )
    print("Load model:", pretrained)

    policy = LowLevelPolicy(
        n_states=n_states,
        n_actions=n_actions,
        action_bounds=action_bounds,
        pretrained=pretrained,
    )
    policy.eval()

    return env, policy



def set_velocity_task(env, target_vel):
    """
    强制设置 cheetah 的 velocity 任务
    target_vel < 0 → backward_vel
    target_vel > 0 → forward_vel
    """
    if target_vel < 0:
        base_task_name = "backward_vel"
        spec = -(target_vel)  
    else:
        base_task_name = "forward_vel"
        spec = target_vel     # 正数

    # 将 v 映射到 [0,1] 的 specification（befitting env.sample_task）
    low, high = env.config["max_vel"]
    spec = np.clip(spec, low, high)
    spec_norm = (spec - low) / (high - low + 1e-6)

    task_dict = {
        "base_task": base_task_name,
        "specification": float(spec_norm)
    }

    if hasattr(env, "sample_task"):
        env.sample_task(task_dict)   # env.task 会自动更新
    else:
        raise RuntimeError("Env has no sample_task(), please verify env implementation.")

    print(f"Set task: {base_task_name}, target_vel={target_vel:.2f}, spec_norm={spec_norm:.3f}")


def rollout_and_render(env, policy, out_video, max_length=500, fps=30):
    """执行 rollout 并保存为视觉化视频"""
    obs, _ = env.reset()

    # ---- 你想测试的目标速度（可以调）----
    target_vel = np.random.uniform(-3.0, 3.0)
    if -0.5 < target_vel < 0.5:
        # 避免太接近 0，不利于测试
        target_vel = 1.5

    set_velocity_task(env, target_vel)

    frames = []
    prev_x = env.get_body_com("torso")[0]

    for t in range(max_length):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        # env.task 是低层训练时的任务向量（例如 [v*] 或 one-hot）
        task_vec = env.task
        task_tensor = torch.as_tensor(task_vec, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = policy.get_action(obs_tensor, task_tensor)
            action = action.cpu().numpy().squeeze(0)

        obs, reward, done, _, _ = env.step(action)

        # ---- 计算当前速度 ----
        xpos = env.get_body_com("torso")[0]
        forward_vel = (xpos - prev_x) / env.dt
        prev_x = xpos

        # ---- 渲染 ----
        frame = env.sim.render(width=640, height=480, camera_name="track")
        frame = np.flipud(frame)
        frame = np.ascontiguousarray(frame)

        # ---- 在画面上写文字 ----
        y = 25
        cv2.putText(frame, f"Step: {t}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y += 30
        cv2.putText(frame, f"Reward: {reward:.3f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y += 30
        cv2.putText(frame, f"Target velocity: {target_vel:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        y += 30
        cv2.putText(frame, f"Current velocity: {forward_vel:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2)

        frames.append(frame)

        if done:
            print("[TERMINATE] Agent fell or done.")
            break

    # ---- 保存视频 ----
    if frames:
        imageio.mimsave(out_video, frames, fps=fps)
        print(f"[OK] Video saved → {out_video}")
    else:
        print("[WARN] No frames captured!")


def main():
    env, policy = load_env_and_policy()
    rollout_and_render(env, policy, OUT_VIDEO, MAX_PATH_LENGTH, FPS)


if __name__ == "__main__":
    main()
