import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# 如果你在项目根目录跑，这两行再打开：
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "submodules/Meta_RL"))

from smrl.experiment.experiment_setup import setup_experiment
from configs.base_configuration import config
from configs.environment_factory import toy1d_vel

# ==== 配置 ====
snapshot_dir = "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/submodules/Meta_RL/data/experiments_thesis/step1_biggerNN_velocity_-12_12_v1_two_directions/_2025-11-07_13-24-59"  # 改成你自己的路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_steps = 250
goal_velocities = [-4, -3, 1, 3, 4]  # 你想测试的目标速度

# ==== 加载环境 ====
config['environment_factory'] = toy1d_vel
expl_env, eval_env = toy1d_vel()

# ==== 自动找最新的 .pkl 模型文件 ====
snapshot_files = [f for f in os.listdir(snapshot_dir) if f.startswith("itr_") and f.endswith(".pkl")]
if len(snapshot_files) == 0:
    raise FileNotFoundError(f"在目录中未找到任何 itr_*.pkl 文件: {snapshot_dir}")

# 按数字排序，取最后一个（最新）
snapshot_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
latest_snapshot = snapshot_files[-1]
policy_path = os.path.join(snapshot_dir, latest_snapshot)

print(f"加载模型文件: {policy_path}")

# ==== 加载 snapshot 并抽取 policy ====
policy_data = torch.load(policy_path, map_location=device)

if isinstance(policy_data, dict):
    if 'policy' in policy_data:
        policy = policy_data['policy']
        print("从 snapshot 中读取键 'policy'")
    elif 'algorithm' in policy_data:
        # 有些版本会把 policy 放在 algorithm 里，这里取 exploration_policy（训练时用的那个）
        algo = policy_data['algorithm']
        # 保险一点：优先用 exploration_policy，没有就用 eval_policy 或 policy
        if hasattr(algo, 'exploration_policy'):
            policy = algo.exploration_policy
            print("从 snapshot 中读取 algorithm.exploration_policy")
        elif hasattr(algo, 'eval_policy'):
            policy = algo.eval_policy
            print("从 snapshot 中读取 algorithm.eval_policy")
        elif hasattr(algo, 'policy'):
            policy = algo.policy
            print("从 snapshot 中读取 algorithm.policy")
        else:
            raise AttributeError("algorithm 对象里找不到 policy / exploration_policy / eval_policy")
    else:
        # 有些 checkpoint 会把对象放在带前缀的键里，例如:
        # 'trainer/Policy trainer/policy', 或者其它带路径的键。我们尝试从键名中搜索包含 'policy' 的项。
        policy_key = None
        for k in policy_data.keys():
            if not isinstance(k, str):
                continue
            kn = k.lower()
            # 忽略包含 qf、encoder、decoder 的 key（这些不是策略本身）
            if 'qf' in kn or 'encoder' in kn or 'decoder' in kn:
                continue
            # 如果 key 以 '/policy' 结尾，或者 key 本身就是 'policy'，认定为策略
            if kn.endswith('/policy') or kn == 'policy' or kn.endswith(' policy') or kn.endswith('policy'):
                policy_key = k
                break

        if policy_key is not None:
            policy = policy_data[policy_key]
            orig_state = policy
            print(f"从 snapshot 中读取键 '{policy_key}'")
        else:
            raise KeyError(f"无法在 {policy_path} 中找到 policy 对象，可用键：{list(policy_data.keys())}")
else:
    policy = policy_data
    print("snapshot 本身就是一个 policy 对象")

# 如果从 checkpoint 中得到的是一个 state_dict（OrderedDict），需要重建 policy 对象并载入权重
import json
from collections import OrderedDict as _OD

if 'orig_state' in locals() and (isinstance(orig_state, _OD) or (isinstance(orig_state, dict) and all(hasattr(v, 'dtype') for v in orig_state.values()))):
    # 尝试从同一 snapshot 目录的 variant.json 中读取超参数以重建网络
    variant_path = os.path.join(snapshot_dir, 'variant.json')
    if os.path.exists(variant_path):
        with open(variant_path, 'r') as f:
            variant = json.load(f)
        # 读取常用参数（有缺省时回退到 base config）
        latent_dim = int(variant.get('latent_dim', config.get('latent_dim', 1)))
        obs_dim = int(variant.get('observation_dim', config.get('observation_dim', 2)))
        action_dim = int(variant.get('action_dim', config.get('action_dim', 1)))
        policy_hsizes = variant.get('policy_kwargs', {}).get('hidden_sizes', config.get('policy_kwargs', {}).get('hidden_sizes'))
    else:
        # 回退到 config 中的值
        latent_dim = config.get('latent_dim', 1)
        obs_dim = config.get('observation_dim', 2)
        action_dim = config.get('action_dim', 1)
        policy_hsizes = config.get('policy_kwargs', {}).get('hidden_sizes')

    # policy 输入的 obs_dim 实际为 env_obs_dim + latent (训练时这样构造)
    policy_input_obs_dim = int(obs_dim + latent_dim)

    # 从 base configuration 中获取 policy 类
    PolicyClass = config.get('policy_type')
    if PolicyClass is None:
        # 作为最后手段，尝试从 rlkit 的 TanhGaussianPolicy 构造（兼容老代码）
        try:
            from rlkit.torch.sac.policies import TanhGaussianPolicy as PolicyClass
        except Exception:
            raise RuntimeError('找不到 policy 类，请在 config 中指定 policy_type 或安装 rlkit')

    # 构造 policy 实例（不同实现的参数名可能不同；MetaRLTanhGaussianPolicy 的签名是 (obs_dim, encoding_dim, action_dim, hidden_sizes=...))
    try:
        policy = PolicyClass(obs_dim=policy_input_obs_dim, encoding_dim=latent_dim, action_dim=action_dim, hidden_sizes=policy_hsizes)
    except TypeError:
        # 备选签名（某些实现可能用位置参数）
        try:
            policy = PolicyClass(policy_input_obs_dim, latent_dim, action_dim, policy_hsizes)
        except Exception as e:
            raise RuntimeError(f'无法实例化 policy 类 {PolicyClass}: {e}')

    # 将 state_dict 载入到 model
    try:
        policy.load_state_dict(orig_state)
    except Exception as e:
        raise RuntimeError(f"载入 state_dict 到 policy 失败: {e}")

    print('已根据 snapshot 的 variant.json 重建 policy 并载入权重。')

policy.to(device)
policy.eval()
print("Policy 已进入 eval 模式。")

# ==== 评估函数 ====
def rollout(env, goal_velocity, render=False):
    # 直接设置当前 task（简单粗暴但够用）
    env._task = {'id': 0, 'goal_velocity': np.array([goal_velocity])}
    obs, _ = env.reset()
    velocities, rewards = [], []
    for t in range(num_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            # 这里假设 latent_dim = 1，所以传一个 shape (1,1) 的零向量作为 encoding
            action, _ = policy.get_action(obs_tensor.cpu().numpy(), np.zeros((1, 1)))
        obs, reward, done, _, info = env.step(action)
        velocities.append(info['velocity'])
        rewards.append(reward)
    return np.array(velocities), np.array(rewards)

# ==== 跑多个目标速度 ====
plt.figure(figsize=(8, 5))
for gv in goal_velocities:
    v, r = rollout(eval_env, gv)
    plt.plot(v, label=f"Goal {gv}")

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Time steps")
plt.ylabel("Velocity")
plt.title("Velocity tracking curves")
plt.legend()
plt.tight_layout()
plt.show()
