"""
Render Cheetah agent (goal tracking) using multi-task trained weights
and save rollout video for checking pure low-level goal tracking ability.

This is the multi-task version of your Ant single-task goal evaluator.
"""

import os
import torch
import imageio
import numpy as np
import cv2

# ========== TODO: 根据你项目实际路径修改 ==========
from third_party.SAC.sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from third_party.SAC.sac_envs.hopper_multi import HopperMulti
from third_party.SAC.model import PolicyNetwork as LowLevelPolicy
# =============================================

# ========== CONFIG ==========
EXPERIMENTS_REPO = "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/low_level_policy"
EXPERIMENT_NAME  = "hopper_multi"   # ⚠️ 改成你的 cheetah multi-task 实验目录
EPOCH            = 3000                                                # 想看的 epoch
OUT_VIDEO        = f"hopper_multitask_goal_epoch{EPOCH}.mp4"
MAX_PATH_LENGTH  = 500
FPS              = 30
# ============================


def load_env_and_policy():
    """加载 Cheetah multi-task 环境与对应的低层策略权重"""

    # ---- 读取 config.json（如果存在）----
    config_path = os.path.join(EXPERIMENTS_REPO, EXPERIMENT_NAME, "config.json")
    env_config = {}
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            env_config = json.load(f)
    print("Loaded config from:", config_path)

    # ---- 创建 Cheetah multi-task 环境 ----
    env = HopperMulti(env_config, render_mode="rgb_array")

    # ---- 获取状态/动作空间维度 ----
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]

    # ---- 加载 policy 权重 ----
    pretrained = os.path.join(
        EXPERIMENTS_REPO,
        EXPERIMENT_NAME,
        "models",
        "policy_model",
        f"epoch_{EPOCH}.pth",
    )
    print("Load model", pretrained)

    policy = LowLevelPolicy(
        n_states=n_states,
        n_actions=n_actions,
        action_bounds=action_bounds,
        pretrained=pretrained,
    )
    policy.eval()

    return env, policy


def set_goal_task(env, target_x):
    """
    强制设置 Cheetah 的 goal tracking 任务。

    约定：
      target_x < 0 → goal_back
      target_x > 0 → goal_front

    注意：这里假设 env.config['max_goal'] 和 env.config['tasks'] 里存在
          'goal_back' / 'goal_front'，和你之前 Ant 的命名保持一致。
          如果你在 Cheetah 里用的是别的名字，比如 'backward_goal' / 'forward_goal'，
          把 base_task_name 换成对应的 key 即可。
    """
    if target_x < 0:
        base_task_name = "goal_back"
        spec_abs = -target_x
    else:
        base_task_name = "goal_front"
        spec_abs = target_x

    # 将目标位置映射到 [0,1] 的 specification，配合 env.sample_task 使用
    low, high = env.config["max_goal"]  # 例如 [2, 10]，和你原来的配置保持一致
    spec_abs = np.clip(spec_abs, low, high)
    spec_norm = (spec_abs - low) / (high - low + 1e-6)

    task_dict = {
        "base_task": base_task_name,
        "specification": float(spec_norm),
    }

    if hasattr(env, "sample_task"):
        env.sample_task(task_dict)   # 内部会设置 env.task / env.base_task
    else:
        raise RuntimeError("Env has no sample_task(task=...) method, 请检查 HalfCheetahMixtureEnv 实现。")

    print(f"Set goal task: {base_task_name}, target_x={target_x:.2f}, spec_norm={spec_norm:.3f}")


def rollout_and_render(env, policy, out_video, max_length=500, fps=30):
    """在 Mujoco 环境中 rollout，并将轨迹渲染为视频（goal tracking）"""

    obs, _ = env.reset()

    # ---- 随机采一个目标位置（你可以改成固定值方便对比）----
    # 比如 Cheetah 从原点出发，正向 [6,10] 米 或 负向 [-10,-6] 米
    if np.random.rand() < 0.5:
        target_x = np.random.uniform(1.0, 10.0)    # 向前
    else:
        target_x = np.random.uniform(-10.0, -1.0)  # 向后

    set_goal_task(env, target_x)

    frames = []

    for t in range(max_length):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        # 低层训练时用的任务向量（可能是 one-hot，也可能是 [goal, vel] 形式）
        # 这里直接从 env.task 取，保证和训练时维度一致，避免 task_dim 不匹配问题。
        if hasattr(env, "task"):
            task_vec = env.task
        else:
            task_vec = np.zeros((1, 1))   # fallback：极端情况下给个占位
        task_tensor = torch.as_tensor(task_vec, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = policy.get_action(obs_tensor, task_tensor)
            action = action.cpu().numpy().squeeze(0)

        obs, reward, done, _, _ = env.step(action)

        # 当前 torso 的 x 坐标（作为位置）
        x_now = env.get_body_com("torso")[0]

        # ---- 渲染一帧图像 ----
        frame = env.sim.render(width=640, height=480, camera_name="track")
        frame = np.flipud(frame)
        frame = np.ascontiguousarray(frame)

        # ---- 在画面上叠加文字信息 ----
        y = 25
        cv2.putText(frame, f"Step: {t}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y += 30
        cv2.putText(frame, f"Reward: {reward:.3f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y += 30
        cv2.putText(frame, f"Target X: {target_x:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        y += 30
        cv2.putText(frame, f"Current X: {x_now:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2)
        y += 30
        cv2.putText(frame, f"Task vec: {task_vec}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        frames.append(frame)

        if done:
            print("[TERMINATE] Agent done or fell.")
            break

    # ---- 保存视频 ----
    if frames:
        imageio.mimsave(out_video, frames, fps=fps)
        print(f"[OK] 视频已保存到 {out_video}")
    else:
        print("[WARN] 没有捕获到任何渲染帧，请检查 env.render() / Mujoco 是否正常。")


def main():
    env, policy = load_env_and_policy()
    rollout_and_render(env, policy, OUT_VIDEO, MAX_PATH_LENGTH, FPS)


if __name__ == "__main__":
    main()
