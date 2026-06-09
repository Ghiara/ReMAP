#!/usr/bin/env python3
"""
Render cheetah multitask videos for baseline policies: PEARL, RL2, MELTS, and CEMRL.

The script auto-resolves the experiment directories from the provided roots,
loads the saved checkpoints, adapts on the same task for a short warmup phase,
and then records the adapted rollout as an MP4.
"""

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

try:
    from PIL import Image
except ImportError:
    Image = None

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SAC_ROOT = ROOT / "submodules" / "SAC"
EXTRA_PATHS = [
    ROOT,
    SAC_ROOT,
    ROOT / "submodules" / "meta_rand_envs",
    ROOT / "submodules" / "rand_param_envs",
    ROOT / "submodules" / "Meta_RL" / "meta-environments-main",
    ROOT / "submodules" / "Meta_RL" / "submodules" / "meta-environments-main",
]
for extra_path in EXTRA_PATHS:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from third_party.rlkit.envs import ENVS
from third_party.rlkit.envs.wrappers import NormalizedBoxEnv
from third_party.rlkit.torch.networks import MlpEncoder, RecurrentEncoder
from third_party.rlkit.torch.rl2.rl2_agent import RL2Agent
from third_party.rlkit.torch.rl2.networks import LSTMPolicy
from third_party.rlkit.torch.sac.agent import PEARLAgent
from third_party.rlkit.torch.sac.policies import TanhGaussianPolicy
from third_party.tigr.agent_module import Agent as TigrAgent


DEVICE = torch.device("cpu")
DEFAULT_OUTPUT_DIR = ROOT / "output" / "baseline_render_videos"

TASK_GROUPS = {
    "goal_backward": [-9.02, -5.10, -3.14],
    "goal_forward": [3.14, 5.10, 9.02],
    "vel_backward": [-2.35, -1.75, -1.45],
    "vel_forward": [1.45, 1.75, 2.35],
}

TRACKED_METRIC = {
    "goal_forward": "position",
    "goal_backward": "position",
    "vel_forward": "velocity",
    "vel_backward": "velocity",
}

TASK_INFO = {
    "vel_forward": {
        "base_task": 0,
        "env_task_key": "forward_vel",
        "color": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    },
    "vel_backward": {
        "base_task": 1,
        "env_task_key": "backward_vel",
        "color": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    },
    "goal_forward": {
        "base_task": 2,
        "env_task_key": "goal_front",
        "color": np.array([1.0, 1.0, 0.0], dtype=np.float32),
    },
    "goal_backward": {
        "base_task": 3,
        "env_task_key": "goal_back",
        "color": np.array([0.0, 1.0, 1.0], dtype=np.float32),
    },
}

PEARL_DEFAULT_ROOT = ROOT / "output" / "pearl_baseline_run1"
RL2_DEFAULT_ROOT = ROOT / "output" / "RL2_baseline_run1"
MELTS_DEFAULT_ROOT = ROOT / "output" / "melts_baseline"
CEMRL_DEFAULT_ROOT = ROOT / "output" / "cemrl_baseline_run1"


def parse_args():
    parser = argparse.ArgumentParser(description="Render cheetah videos for PEARL, RL2, MELTS, and CEMRL baselines.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=["pearl", "rl2", "melts", "cemrl"],
        default=["pearl", "rl2", "melts", "cemrl"],
        help="Which baselines to render.",
    )
    parser.add_argument("--pearl-root", type=Path, default=PEARL_DEFAULT_ROOT)
    parser.add_argument("--rl2-root", type=Path, default=RL2_DEFAULT_ROOT)
    parser.add_argument("--melts-root", type=Path, default=MELTS_DEFAULT_ROOT)
    parser.add_argument("--cemrl-root", type=Path, default=CEMRL_DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--warmup-trajs",
        type=int,
        default=1,
        help="How many same-task adaptation trajectories to run before recording the video.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def deep_update_dict(source, target):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update_dict(value, target[key])
        else:
            target[key] = value
    return target


def npify_dict(data):
    if isinstance(data, dict):
        return {k: npify_dict(v) for k, v in data.items()}
    if isinstance(data, list):
        return [npify_dict(v) for v in data]
    return np.asarray(data) if isinstance(data, (list, tuple)) else data


def resolve_experiment_dir(root: Path) -> Path:
    root = root.resolve()
    if (root / "weights").is_dir() and (root / "variant.json").is_file():
        return root

    candidates = sorted(
        {
            path.parent
            for path in root.rglob("variant.json")
            if (path.parent / "weights").is_dir()
        }
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find an experiment directory with weights under {root}")
    return candidates[-1]


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def set_cheetah_task(env, task_name: str, target_value: float):
    raw = env._wrapped_env if hasattr(env, "_wrapped_env") else env
    task = TASK_INFO[task_name]
    target_value = float(target_value)

    raw._task = {
        "base_task": task["base_task"],
        "specification": target_value,
        "color": task["color"],
    }
    raw.base_task = task["base_task"]
    raw.task_specification = target_value
    raw._goal = target_value

    # The cheetah reward reads from raw.task[raw.base_task] and raw.norm.
    # Keep those in sync with the requested render target as well.
    env_task_key = task.get("env_task_key")
    if hasattr(raw, "config") and isinstance(getattr(raw, "config", None), dict):
        env_task_id = raw.config.get("tasks", {}).get(env_task_key, task["base_task"])
        raw.base_task = env_task_id
        if hasattr(raw, "task") and raw.task is not None:
            task_arr = np.asarray(raw.task, dtype=np.float32).copy()
            needed = int(max(env_task_id + 1, task_arr.shape[0]))
            if task_arr.shape[0] < needed:
                padded = np.zeros(needed, dtype=np.float32)
                padded[:task_arr.shape[0]] = task_arr
                task_arr = padded
            task_arr[:] = 0.0
            task_arr[env_task_id] = target_value
            raw.task = task_arr
        raw.norm = target_value

    if hasattr(raw, "update_base_task"):
        try:
            raw.update_base_task(raw.base_task)
        except Exception:
            pass
    if hasattr(raw, "update_task") and hasattr(raw, "task"):
        try:
            raw.update_task(raw.task)
        except Exception:
            pass

    try:
        raw.color = task["color"]
        raw.recolor()
    except Exception:
        pass
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        next_obs, reward, done, info = result
    if isinstance(next_obs, tuple):
        next_obs = next_obs[0]
    return next_obs, float(reward), bool(done), info


def capture_frame(env, width: int, height: int):
    raw = env._wrapped_env if hasattr(env, "_wrapped_env") else env
    if hasattr(raw, "get_image"):
        return raw.get_image(width=width, height=height)[::-1]
    frame = raw.render(mode="rgb_array", width=width, height=height)
    if frame is None:
        raise RuntimeError("render() returned None")
    return frame


def get_metric_value(obs: np.ndarray, task_name: str) -> float:
    if TRACKED_METRIC[task_name] == "velocity":
        return float(obs[8])
    return float(obs[17])


def add_overlay(frame, baseline_name: str, task_name: str, target_value: float, metric_value: float):
    frame = np.ascontiguousarray(frame[:, :, :3])
    if cv2 is None:
        return frame

    color = (255, 0, 0)
    metric_name = TRACKED_METRIC[task_name]
    err = abs(metric_value - target_value)
    lines = [
        f"baseline: {baseline_name}",
        f"task: {task_name}",
        f"target {metric_name}: {target_value:+.2f}",
        f"current {metric_name}: {metric_value:+.2f}",
        f"error: {err:.2f}",
    ]
    for idx, text in enumerate(lines):
        y = 34 + idx * 20
        cv2.putText(frame, text, (29, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame


def save_animation(out_path: Path, frames, fps: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        imageio.mimsave(out_path, frames, fps=fps)
        return out_path
    if Image is not None:
        gif_path = out_path.with_suffix('.gif')
        pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
        duration_ms = max(int(1000 / max(fps, 1)), 1)
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return gif_path
    raise RuntimeError('Neither imageio nor PIL is available, cannot save animation output.')


def write_summary(rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    headers = [
        "baseline",
        "task",
        "target_value",
        "num_steps",
        "final_value",
        "mean_abs_error",
        "tail_mean_abs_error",
        "video_path",
    ]
    with summary_path.open("w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")
    return summary_path


def build_pearl_env_agent(exp_dir: Path):
    variant = json.loads((exp_dir / "variant.json").read_text())
    env = NormalizedBoxEnv(ENVS[variant["env_name"]](**variant["env_params"]))
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    latent_dim = variant["latent_size"]
    ctx_input_dim = obs_dim + action_dim + reward_dim
    if variant["algo_params"]["use_next_obs_in_context"]:
        ctx_input_dim += obs_dim
    ctx_output_dim = latent_dim * 2 if variant["algo_params"]["use_information_bottleneck"] else latent_dim
    encoder_cls = RecurrentEncoder if variant["algo_params"]["recurrent"] else MlpEncoder
    context_encoder = encoder_cls(hidden_sizes=[200, 200, 200], input_size=ctx_input_dim, output_size=ctx_output_dim)
    policy = TanhGaussianPolicy(
        hidden_sizes=[variant["net_size"], variant["net_size"], variant["net_size"]],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(latent_dim, context_encoder, policy, **variant["algo_params"])
    weights_dir = exp_dir / "weights"
    context_encoder.load_state_dict(safe_torch_load(weights_dir / "context_encoder.pth"))
    policy.load_state_dict(safe_torch_load(weights_dir / "policy.pth"))
    agent.to(DEVICE)
    agent.eval()
    return env, agent, variant


def build_rl2_env_agent(exp_dir: Path):
    variant = json.loads((exp_dir / "variant.json").read_text())
    env = NormalizedBoxEnv(ENVS[variant["env_name"]](**variant["env_params"]))
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    net_size = variant.get("net_size", 300)

    policy = LSTMPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_size=net_size, num_layers=1)
    weights_dir = exp_dir / "weights"
    policy.load_state_dict(safe_torch_load(weights_dir / "policy.pth"))
    agent = RL2Agent(obs_dim=obs_dim, action_dim=action_dim, hidden_size=net_size, num_layers=1, latent_dim=net_size)
    agent.policy.load_state_dict(policy.state_dict())
    agent.policy.to(DEVICE)
    agent.eval()
    return env, agent, variant


def build_melts_env_agent(exp_dir: Path):
    variant = json.loads((exp_dir / "variant.json").read_text())
    env_params = copy.deepcopy(variant["env_params"])
    use_normalized = env_params.pop("use_normalized_env", True)
    env = ENVS[variant["env_name"]](**env_params)
    if use_normalized:
        env = NormalizedBoxEnv(env)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    latent_dim = variant["algo_params"]["latent_size"]
    time_steps = variant["algo_params"]["time_steps"]
    num_classes = variant["reconstruction_params"]["num_classes"]
    net_complex = variant["reconstruction_params"]["net_complex_enc_dec"]
    encoder_type = variant["algo_params"]["encoder_type"]
    timestep_combination = variant["algo_params"]["timestep_combination"]
    encoding_mode = variant["algo_params"]["encoding_mode"]
    if encoder_type == "gru" and encoding_mode != "trajectory":
        encoding_mode = "trajectory"
    elif encoder_type in {"transformer", "conv"} and encoding_mode != "transitionSharedY":
        encoding_mode = "transitionSharedY"
    if encoding_mode == "transitionSharedY":
        encoder_input_dim = obs_dim + reward_dim + obs_dim
        shared_dim = int(encoder_input_dim * net_complex)
    else:
        encoder_input_dim = time_steps * (obs_dim + reward_dim + obs_dim)
        shared_dim = int((encoder_input_dim / time_steps) * net_complex)

    if variant["inference_option"] == "dpmm":
        from third_party.tigr.task_inference.dpmm_bnp import BNPModel
        from third_party.tigr.task_inference.dpmm_inference import DecoupledEncoder

        dp = variant.get("dpmm_params", {})
        bnp_model = BNPModel(
            save_dir=tempfile.mkdtemp(),
            start_epoch=int(1e12),
            gamma0=dp.get("gamma0", 5.0),
            num_lap=dp.get("num_lap", 2),
            fit_interval=dp.get("fit_interval", "epoch"),
            kl_method=dp.get("kl_method", "soft"),
            birth_kwargs=dp.get("birth_kwargs", {}),
            merge_kwargs=dp.get("merge_kwargs", {}),
        )
    else:
        from third_party.tigr.task_inference.true_gmm_inference import DecoupledEncoder

        bnp_model = None

    encoder = DecoupledEncoder(
        shared_dim,
        encoder_input_dim,
        latent_dim,
        num_classes,
        time_steps,
        encoding_mode=encoding_mode,
        timestep_combination=timestep_combination,
        encoder_type=encoder_type,
        bnp_model=bnp_model,
    )
    policy_latent_dim = latent_dim if variant["algo_params"]["sac_context_type"] == "sample" else latent_dim * 2
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + policy_latent_dim,
        action_dim=action_dim,
        latent_dim=policy_latent_dim,
        hidden_sizes=[variant["algo_params"]["sac_layer_size"]] * 3,
    )

    weights_dir = exp_dir / "weights"
    itr = max(int(path.stem.split("_")[-1]) for path in weights_dir.glob("encoder_itr_*.pth"))
    encoder.load_state_dict(safe_torch_load(weights_dir / f"encoder_itr_{itr}.pth"))
    policy.load_state_dict(safe_torch_load(weights_dir / f"policy_itr_{itr}.pth"))
    agent = TigrAgent(encoder, policy, use_sample=(variant["algo_params"]["sac_context_type"] == "sample"))
    agent.to(DEVICE)
    agent.eval()

    stats_dict = None
    stats_path = weights_dir / "stats_dict.json"
    if variant["algo_params"].get("use_data_normalization", False) and stats_path.exists():
        stats_dict = npify_dict(json.loads(stats_path.read_text()))

    info = {
        "obs_dim": obs_dim,
        "time_steps": time_steps,
        "encoding_mode": encoding_mode,
        "use_data_normalization": variant["algo_params"].get("use_data_normalization", False),
        "stats_dict": stats_dict,
        "max_path_length": variant["algo_params"]["max_path_length"],
    }
    return env, agent, info


def build_cemrl_env_agent(exp_dir: Path):
    return build_melts_env_agent(exp_dir)


def pearl_warmup(env, agent, task_name, target_value, max_steps, warmup_trajs):
    set_cheetah_task(env, task_name, target_value)
    agent.clear_z()
    for _ in range(warmup_trajs):
        obs = set_cheetah_task(env, task_name, target_value)
        for _step in range(max_steps):
            (action, _), _ = agent.get_action(obs, deterministic=True)
            next_obs, reward, done, info = step_env(env, action)
            agent.update_context((obs, action, reward, next_obs, done, info))
            obs = next_obs
            if done:
                break
        if agent.context is not None:
            agent.infer_posterior(agent.context)


def rl2_warmup(env, agent, task_name, target_value, max_steps, warmup_trajs):
    set_cheetah_task(env, task_name, target_value)
    agent.clear_z()
    for _ in range(warmup_trajs):
        obs = set_cheetah_task(env, task_name, target_value)
        for _step in range(max_steps):
            (action, _), _ = agent.get_action(obs, deterministic=True)
            next_obs, reward, done, info = step_env(env, action)
            agent.update_context((obs, action, reward, next_obs, done, info))
            obs = next_obs
            if done:
                break


def init_melts_context(info):
    enc_input_dim = info["obs_dim"] + 1 + info["obs_dim"]
    return torch.zeros(1, info["time_steps"], enc_input_dim, device=DEVICE)


def normalize_melts_transition(obs, reward, next_obs, stats_dict):
    obs_norm = obs
    reward_norm = np.array([reward], dtype=np.float32)
    next_obs_norm = next_obs
    if stats_dict is not None:
        try:
            obs_norm = (obs - stats_dict["observations"]["mean"]) / (stats_dict["observations"]["std"] + 1e-8)
            reward_norm = (reward_norm - stats_dict["rewards"]["mean"]) / (stats_dict["rewards"]["std"] + 1e-8)
            next_obs_norm = (next_obs - stats_dict["next_observations"]["mean"]) / (stats_dict["next_observations"]["std"] + 1e-8)
        except Exception:
            pass
    return obs_norm, reward_norm, next_obs_norm


def update_melts_context(context, obs, reward, next_obs, info):
    stats_dict = info["stats_dict"] if info["use_data_normalization"] else None
    obs_norm, reward_norm, next_obs_norm = normalize_melts_transition(obs, reward, next_obs, stats_dict)
    transition = np.concatenate(
        [
            np.asarray(obs_norm, dtype=np.float32),
            np.asarray(reward_norm, dtype=np.float32).reshape(1),
            np.asarray(next_obs_norm, dtype=np.float32),
        ]
    )
    transition_t = torch.from_numpy(transition).to(DEVICE).view(1, 1, -1)
    context = torch.roll(context, shifts=-1, dims=1)
    context[:, -1:, :] = transition_t
    return context


def format_melts_encoder_input(context, info):
    if info["encoding_mode"] == "trajectory":
        return context.view(1, -1)
    return context


def melts_warmup(env, agent, info, task_name, target_value, max_steps, warmup_trajs):
    context = init_melts_context(info)
    for _ in range(warmup_trajs):
        obs = set_cheetah_task(env, task_name, target_value)
        for _step in range(max_steps):
            enc_input = format_melts_encoder_input(context, info)
            with torch.no_grad():
                (action, _), _ = agent.get_action(enc_input, obs, deterministic=True)
            next_obs, reward, done, _ = step_env(env, action)
            context = update_melts_context(context, obs, reward, next_obs, info)
            obs = next_obs
            if done:
                break
    return context


def render_pearl_video(env, agent, task_name, target_value, out_path, max_steps, fps, width, height, warmup_trajs):
    pearl_warmup(env, agent, task_name, target_value, max_steps, warmup_trajs)
    obs = set_cheetah_task(env, task_name, target_value)
    frames = []
    metric_trace = []
    for _step in range(max_steps):
        prev_obs = obs
        (action, _), _ = agent.get_action(prev_obs, deterministic=True)
        obs, reward, done, info = step_env(env, action)
        agent.update_context((prev_obs, action, reward, obs, done, info))
        metric_value = get_metric_value(obs, task_name)
        metric_trace.append(metric_value)
        frame = add_overlay(capture_frame(env, width, height), "pearl", task_name, target_value, metric_value)
        frames.append(frame)
        if done:
            break
    saved_path = save_animation(out_path, frames, fps)
    return summarize_video("pearl", task_name, target_value, saved_path, metric_trace)


def render_rl2_video(env, agent, task_name, target_value, out_path, max_steps, fps, width, height, warmup_trajs):
    rl2_warmup(env, agent, task_name, target_value, max_steps, warmup_trajs)
    obs = set_cheetah_task(env, task_name, target_value)
    frames = []
    metric_trace = []
    for _step in range(max_steps):
        (action, _), _ = agent.get_action(obs, deterministic=True)
        next_obs, reward, done, info = step_env(env, action)
        agent.update_context((obs, action, reward, next_obs, done, info))
        obs = next_obs
        metric_value = get_metric_value(obs, task_name)
        metric_trace.append(metric_value)
        frame = add_overlay(capture_frame(env, width, height), "rl2", task_name, target_value, metric_value)
        frames.append(frame)
        if done:
            break
    saved_path = save_animation(out_path, frames, fps)
    return summarize_video("rl2", task_name, target_value, saved_path, metric_trace)


def render_melts_video(env, agent, info, task_name, target_value, out_path, max_steps, fps, width, height, warmup_trajs):
    context = melts_warmup(env, agent, info, task_name, target_value, max_steps, warmup_trajs)
    obs = set_cheetah_task(env, task_name, target_value)
    frames = []
    metric_trace = []
    for _step in range(max_steps):
        enc_input = format_melts_encoder_input(context, info)
        with torch.no_grad():
            (action, _), _ = agent.get_action(enc_input, obs, deterministic=True)
        next_obs, reward, done, _ = step_env(env, action)
        context = update_melts_context(context, obs, reward, next_obs, info)
        obs = next_obs
        metric_value = get_metric_value(obs, task_name)
        metric_trace.append(metric_value)
        frame = add_overlay(capture_frame(env, width, height), "melts", task_name, target_value, metric_value)
        frames.append(frame)
        if done:
            break
    saved_path = save_animation(out_path, frames, fps)
    return summarize_video("melts", task_name, target_value, saved_path, metric_trace)


def render_cemrl_video(env, agent, info, task_name, target_value, out_path, max_steps, fps, width, height, warmup_trajs):
    context = melts_warmup(env, agent, info, task_name, target_value, max_steps, warmup_trajs)
    obs = set_cheetah_task(env, task_name, target_value)
    frames = []
    metric_trace = []
    for _step in range(max_steps):
        enc_input = format_melts_encoder_input(context, info)
        with torch.no_grad():
            (action, _), _ = agent.get_action(enc_input, obs, deterministic=True)
        next_obs, reward, done, _ = step_env(env, action)
        context = update_melts_context(context, obs, reward, next_obs, info)
        obs = next_obs
        metric_value = get_metric_value(obs, task_name)
        metric_trace.append(metric_value)
        frame = add_overlay(capture_frame(env, width, height), "cemrl", task_name, target_value, metric_value)
        frames.append(frame)
        if done:
            break
    saved_path = save_animation(out_path, frames, fps)
    return summarize_video("cemrl", task_name, target_value, saved_path, metric_trace)


def summarize_video(baseline, task_name, target_value, out_path, metric_trace):
    trace = np.asarray(metric_trace, dtype=np.float32)
    final_value = float(trace[-1]) if len(trace) else float("nan")
    mean_abs_error = float(np.mean(np.abs(trace - target_value))) if len(trace) else float("nan")
    tail = trace[-50:] if len(trace) >= 50 else trace
    tail_mean_abs_error = float(np.mean(np.abs(tail - target_value))) if len(tail) else float("nan")
    return {
        "baseline": baseline,
        "task": task_name,
        "target_value": float(target_value),
        "num_steps": int(len(trace)),
        "final_value": final_value,
        "mean_abs_error": mean_abs_error,
        "tail_mean_abs_error": tail_mean_abs_error,
        "video_path": str(out_path),
    }


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    rows = []

    baseline_roots = {
        "pearl": args.pearl_root,
        "rl2": args.rl2_root,
        "melts": args.melts_root,
        "cemrl": args.cemrl_root,
    }

    for baseline in args.baselines:
        exp_dir = resolve_experiment_dir(baseline_roots[baseline])
        print(f"[{baseline}] experiment dir: {exp_dir}")

        if baseline == "pearl":
            env, agent, variant = build_pearl_env_agent(exp_dir)
            max_steps = min(args.max_steps, variant["algo_params"]["max_path_length"])
        elif baseline == "rl2":
            env, agent, variant = build_rl2_env_agent(exp_dir)
            max_steps = min(args.max_steps, variant["algo_params"]["max_path_length"])
        elif baseline == "melts":
            env, agent, info = build_melts_env_agent(exp_dir)
            max_steps = min(args.max_steps, info["max_path_length"])
        else:
            env, agent, info = build_cemrl_env_agent(exp_dir)
            max_steps = min(args.max_steps, info["max_path_length"])

        baseline_output_dir = args.output_dir / baseline / exp_dir.name
        for task_name, targets in TASK_GROUPS.items():
            for target_value in targets:
                video_name = f"{task_name}_target_{target_value:+.2f}.mp4".replace("+", "plus_").replace("-", "minus_")
                out_path = baseline_output_dir / task_name / video_name
                print(f"  rendering {task_name} @ {target_value:+.2f} -> {out_path}")
                if baseline == "pearl":
                    row = render_pearl_video(
                        env, agent, task_name, target_value, out_path,
                        max_steps, args.fps, args.width, args.height, args.warmup_trajs,
                    )
                elif baseline == "rl2":
                    row = render_rl2_video(
                        env, agent, task_name, target_value, out_path,
                        max_steps, args.fps, args.width, args.height, args.warmup_trajs,
                    )
                elif baseline == "melts":
                    row = render_melts_video(
                        env, agent, info, task_name, target_value, out_path,
                        max_steps, args.fps, args.width, args.height, args.warmup_trajs,
                    )
                else:
                    row = render_cemrl_video(
                        env, agent, info, task_name, target_value, out_path,
                        max_steps, args.fps, args.width, args.height, args.warmup_trajs,
                    )
                rows.append(row)
                print(
                    f"    steps={row['num_steps']} final={row['final_value']:+.3f} "
                    f"mae={row['mean_abs_error']:.3f} tail_mae={row['tail_mean_abs_error']:.3f}"
                )
        env.close()

    summary_path = write_summary(rows, args.output_dir)
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
