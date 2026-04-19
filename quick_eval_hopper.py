"""
快速验证 Hopper 低层策略的关键子任务表现。
只测 goal_forward@6m 和 vel_backward@2.0，1个episode，快速出结论（约1分钟）。

用法：
    python quick_eval_hopper.py            # 测最新 epoch（默认）
    python quick_eval_hopper.py 500        # 测 epoch 500
"""

import os
import sys
import numpy as np
import torch

# ============ 配置 ============
LOW_LEVEL_EXPERIMENTS_REPO = "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/low_level_policy/"
LOW_LEVEL_EXPERIMENT_NAME  = "hopper_multi"

# 从命令行参数读取 epoch，否则自动找最新
if len(sys.argv) > 1:
    LOW_LEVEL_EPOCH = int(sys.argv[1])
else:
    model_dir = os.path.join(LOW_LEVEL_EXPERIMENTS_REPO, LOW_LEVEL_EXPERIMENT_NAME, "models", "policy_model")
    epochs = sorted([int(f.replace("epoch_", "").replace(".pth", ""))
                     for f in os.listdir(model_dir) if f.endswith(".pth")])
    LOW_LEVEL_EPOCH = epochs[-1]
    print(f"[auto] 使用最新 checkpoint: epoch {LOW_LEVEL_EPOCH}")

# 只测最关心的关键子任务
QUICK_TESTS = [
    ("goal_forward", [6.0]),
    ("vel_backward",  [2.0]),
]
NUM_EPISODES = 1
MAX_STEPS    = 300
# ==============================

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submodules/SAC"))

from sac_envs.hopper_multi import HopperMulti
from model import PolicyNetwork as TransferFunction
from train_striding_predictor import get_complex_agent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env():
    import json
    cfg_path = os.path.join(LOW_LEVEL_EXPERIMENTS_REPO, LOW_LEVEL_EXPERIMENT_NAME, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    env_cfg = cfg.get("env_params", cfg.get("config", cfg))
    return HopperMulti(env_cfg)


def load_policy(env):
    policy = get_complex_agent(env, {
        "experiments_repo": LOW_LEVEL_EXPERIMENTS_REPO,
        "experiment_name":  LOW_LEVEL_EXPERIMENT_NAME,
        "epoch": LOW_LEVEL_EPOCH,
    })
    policy.eval()
    return policy


def build_subgoal(env, value, skill_type):
    t = env.config["tasks"]
    simple = np.zeros_like(env.task, dtype=np.float32)
    if   skill_type == "vel_forward":   simple[t["forward_vel"]]  = +value
    elif skill_type == "vel_backward":  simple[t["backward_vel"]] = -value
    elif skill_type == "goal_forward":  simple[t["goal_front"]]   = +value
    elif skill_type == "goal_backward": simple[t["goal_back"]]    = -value
    return torch.from_numpy(simple).float().to(DEVICE)


def evaluate_one(env, policy, skill_type, value):
    phys_target = +value if "forward" in skill_type else -value
    obs, _ = env.reset()
    subgoal = build_subgoal(env, value, skill_type)
    track = []
    for _ in range(MAX_STEPS):
        obs_t = torch.from_numpy(obs).float().to(DEVICE)
        with torch.no_grad():
            action = policy.get_action(obs_t, subgoal, return_dist=False)
        obs, _, done, truncated, _ = env.step(action.cpu().numpy(), healthy_scale=0)
        val = float(env.sim.data.qvel[0] if "vel" in skill_type else env.sim.data.qpos[0])
        track.append(val)
        if done or truncated:
            break
    track = np.array(track)
    mae = np.mean(np.abs(track - phys_target))
    return mae, len(track), track


def main():
    print(f"\n{'='*55}")
    print(f"  Quick Eval  |  experiment: {LOW_LEVEL_EXPERIMENT_NAME}  |  epoch: {LOW_LEVEL_EPOCH}")
    print(f"{'='*55}")

    env = make_env()
    policy = load_policy(env)

    results = []
    for skill_type, values in QUICK_TESTS:
        for v in values:
            mae, tlen, _ = evaluate_one(env, policy, skill_type, v)
            phys = +v if "forward" in skill_type else -v
            status = "✓ GOOD" if mae < (1.5 if "goal" in skill_type else 0.8) else "✗ POOR"
            print(f"  {skill_type:<16} target={phys:+.1f}  MAE={mae:.3f}  len={tlen:3d}  {status}")
            results.append((skill_type, v, mae))

    print(f"{'='*55}\n")
    env.close()


if __name__ == "__main__":
    main()
