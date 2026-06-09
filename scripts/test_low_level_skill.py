import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from third_party.SAC.sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from third_party.SAC.sac_envs.walker_multi import WalkerMulti
from third_party.SAC.sac_envs.hopper_multi import HopperMulti
from third_party.SAC.sac_envs.ant_multi import AntMulti
from third_party.SAC.model import PolicyNetwork as TransferFunction
from task_inference_high_level_cross_agent_deployment import get_complex_agent

# train_striding_predictor_LL_vel_evaluation


#这个脚本是用来测试训练出来的低层技能的。它会加载指定实验的低层策略，并在对应环境中执行预设的技能（如前向速度、后向速度、前向位置、后向位置），记录并绘制跟踪值（如实际速度或位置）随时间的变化，以评估技能的效果。

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 这些常量记得改成你自己的 ====
LOW_LEVEL_EXPERIMENTS_REPO = "/root/bayes-tmp/bowang/Inference-reutilization-MRL/output/low_level_policy/"
LOW_LEVEL_EXPERIMENT_NAME  = "ant_multi_new_config_v3"
LOW_LEVEL_EPOCH            = 500
# ===================================


# ===== 1. 创建环境 =====
def make_env():
    import json
    config_path = os.path.join(
        LOW_LEVEL_EXPERIMENTS_REPO,
        LOW_LEVEL_EXPERIMENT_NAME,
        "config.json"
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    if "env_params" in cfg:
        env_cfg = cfg["env_params"]
    elif "config" in cfg:
        env_cfg = cfg["config"]
    else:
        env_cfg = cfg

    env_name = env_cfg.get("env")
    if env_name == "half_cheetah_multi":
        env = HalfCheetahMixtureEnv(env_cfg)
    elif env_name == "hopper_multi":
        env = HopperMulti(env_cfg)
    elif env_name == "walker_multi":
        env = WalkerMulti(env_cfg)
    elif env_name == "ant_multi":
        env = AntMulti(env_cfg)
        # if not hasattr(env, "config") or env.config is None:
        #     env.config = env_cfg
    else:
        raise ValueError(f"Unsupported env '{env_name}' in {config_path}")

    return env


# ===== 2. 加载策略 =====
def load_low_level_policy(env):
    complex_agent_cfg = {
        "experiments_repo": LOW_LEVEL_EXPERIMENTS_REPO,
        "experiment_name":  LOW_LEVEL_EXPERIMENT_NAME,
        "epoch": LOW_LEVEL_EPOCH,
    }
    policy = get_complex_agent(env, complex_agent_cfg)
    policy.eval()
    return policy


# ===== 3. 构造 subgoal：支持 velocity & goal =====
def build_subgoal(env, value, skill_type):
    """
    value: 目标值模长 (>=0)
    skill_type:
        vel_forward
        vel_backward
        goal_forward
        goal_backward
    """

    tasks_cfg = env.config["tasks"]
    idx_GF = tasks_cfg["goal_front"]
    idx_GB = tasks_cfg["goal_back"]
    idx_FV = tasks_cfg["forward_vel"]
    idx_BV = tasks_cfg["backward_vel"]

    simple = np.zeros_like(env.task, dtype=np.float32)

    if skill_type == "vel_forward":
        simple[idx_FV] = +value

    elif skill_type == "vel_backward":
        simple[idx_BV] = -value                # ← 负号很重要

    elif skill_type == "goal_forward":
        simple[idx_GF] = +value

    elif skill_type == "goal_backward":
        simple[idx_GB] = -value                # ← 负号很重要

    else:
        raise ValueError(f"Unknown skill_type {skill_type}")

    return torch.from_numpy(simple).float().to(DEVICE)


# ===== 4. 获取物理目标值（带符号） =====
def get_physical_target(value, skill_type):
    if "forward" in skill_type:
        return +value
    else:
        return -value


# ===== 5. 滑动平均 =====
def moving_average(x, window=20):
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


# ===== 6. 画单 episode 图 =====
def plot_single_episode(traj, traj_ma, phys_target, skill_label, save_path):
    t = np.arange(len(traj))
    plt.figure(figsize=(10, 4))
    plt.plot(t, traj, label="raw value")
    plt.plot(t, traj_ma, label="moving avg", linewidth=2)
    plt.axhline(y=phys_target, linestyle="--", label=f"target {skill_label}")
    plt.xlabel("step")
    plt.ylabel("tracked value")
    plt.title(f"Skill Test – {skill_label}")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


# ===== 7. 测试某技能 =====
def evaluate_skill(env, policy, value, skill_type,
                   num_episodes=3, max_steps=500, render=False):

    phys_target = get_physical_target(value, skill_type)

    all_track = []
    all_t = []
    global_step = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        simple_subgoal = build_subgoal(env, value, skill_type)

        traj = []

        for t in range(max_steps):
            obs_tensor = torch.from_numpy(obs).float().to(DEVICE)

            with torch.no_grad():
                action = policy.get_action(obs_tensor, simple_subgoal, return_dist=False)

            obs, _, done, truncated, info = env.step(action.cpu().numpy(), healthy_scale=0)

            # ========== 根据技能类型选 tracking 变量 ==========
            if "vel" in skill_type:
                track_val = float(env.sim.data.qvel[0])   # root x velocity across supported mujoco agents
            else:
                track_val = float(env.sim.data.qpos[0])   # root x position across supported mujoco agents

            traj.append(track_val)
            all_track.append(track_val)
            all_t.append(global_step)
            global_step += 1

            if done or truncated:
                break

        traj = np.array(traj)
        traj_ma = moving_average(traj, window=20)

        mae_raw = np.mean(np.abs(traj - phys_target))
        mae_ma  = np.mean(np.abs(traj_ma - phys_target))

        print(f"[{skill_type} | |value|={value}, phys_target={phys_target}, ep={ep}] "
              f"raw MAE={mae_raw:.3f}, smoothed MAE={mae_ma:.3f}, len={len(traj)}")

        plot_single_episode(
            traj, traj_ma, phys_target,
            skill_label=f"{skill_type} target={phys_target:.2f}",
            save_path=f"{LOW_LEVEL_EXPERIMENT_NAME}_plots/{skill_type}_ep{ep}_value{value}.png"
        )

    return np.array(all_t), np.array(all_track)


def main():
    env = make_env()
    policy = load_low_level_policy(env)

    # ========================================================
    # 🌟 重点修改：现在可以为速度和目标设置不同的值列表
    # ========================================================
    # 速度目标值 (e.g., vx=1.0, vx=-2.0)
    vel_values = [1.0, 2.0, 2.5]
    vel_skills = ["vel_forward", "vel_backward"]

    # 位置目标值 (e.g., x=5.0, x=-10.0)
    # 通常位置目标需要更大的数值
    goal_values = [2.0, 6.0, 10.0]
    goal_skills = ["goal_forward", "goal_backward"]

    # 整合所有要测试的技能及其对应的值列表
    tests = [
        (vel_skills, vel_values),
        (goal_skills, goal_values),
    ]
    # ========================================================

    for skills_list, values_list in tests:
        for skill in skills_list:
            print("\n=======================================")
            print(f"====== Testing skill: {skill} ======")
            print("=======================================")

            for v in values_list:
                t, track = evaluate_skill(
                    env, policy, v, skill,
                    num_episodes=3, max_steps=500
                )

                track_ma = moving_average(track, window=50)
                phys_target = get_physical_target(v, skill)

                # 绘制所有 episode 的汇总图
                plt.figure(figsize=(10, 4))
                plt.plot(t, track, alpha=0.3, label="raw data (all episodes)", color='gray')
                plt.plot(t, track_ma, linewidth=3, label="moving avg (all episodes)", color='blue')
                plt.axhline(y=phys_target, linestyle="--", color='red',
                            label=f"Target {phys_target:.2f}")
                plt.xlabel("global step")
                plt.ylabel("tracked value")
                plt.title(f"All episodes – {skill}, Target Value={phys_target}")
                plt.legend()

                os.makedirs(f"{LOW_LEVEL_EXPERIMENT_NAME}_plots", exist_ok=True)
                plt.savefig(f"{LOW_LEVEL_EXPERIMENT_NAME}_plots/{skill}_ALL_value{v}.png", dpi=200)
                plt.close()

    env.close()

if __name__ == "__main__":
    main()



