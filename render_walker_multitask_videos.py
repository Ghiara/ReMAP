#这个版本是使用toy到对应的位置就停止的，并且在接近目标位置时平滑动作以避免抖动。其他环境保持不变，并且都可以用



#运行命令：python render_walker_multitask_videos.py --agent walker 



import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SAC_ROOT = ROOT / "submodules" / "SAC"
if str(SAC_ROOT) not in sys.path:
    sys.path.insert(0, str(SAC_ROOT))

from sac_envs.ant_multi_old import AntMulti
from sac_envs.half_cheetah_multi import HalfCheetahMixtureEnv
from sac_envs.hopper_multi import HopperMulti
from sac_envs.walker_multi import WalkerMulti
from model import PolicyNetwork as TransferFunction
from rlkit.envs import ENVS  # toy patch
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy  # toy patch
from tigr.task_inference.dpmm_inference import DecoupledEncoder  # toy patch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOY_DEVICE = torch.device("cpu")  # toy patch
OUTPUT_DIR = ROOT / "output" / "multitask_videos"
TOY_GOAL_BRAKE_DISTANCE = 0.1  # toy patch: when |position error| is below this, start replacing the policy action with the brake controller
TOY_GOAL_SETTLE_DISTANCE = 0.25  # toy patch: if the toy is within this distance from the goal, we consider the position close enough to stop
TOY_GOAL_SETTLE_VELOCITY = 0.12  # toy patch: if the toy's absolute velocity is below this near the goal, we consider it settled and output zero action
TOY_GOAL_SLOW_DISTANCE = 0.75  # toy patch: start smoothing position-goal actions inside this distance to avoid near-goal jitter
TOY_GOAL_NEAR_MAX_ACTION = 0.20  # toy patch: maximum absolute action right at the goal after smoothing
TOY_GOAL_NEAR_ACTION_DELTA = 0.035  # toy patch: max per-frame action change right at the goal after smoothing

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

TASK_COLORS_BGR = {
    "goal_forward": (0, 215, 255),
    "goal_backward": (255, 255, 0),
    "vel_forward": (80, 80, 255),
    "vel_backward": (80, 220, 80),
}

AGENT_SPECS = {
    "cheetah": {
        "env_cls": HalfCheetahMixtureEnv,
        "default_experiment_root": ROOT / "output" / "low_level_policy" / "cheetah_multitask_new_config_v0",
        "default_max_steps": 400,
        "task_name_map": {
            "goal_forward": "goal_front",
            "goal_backward": "goal_back",
            "vel_forward": "forward_vel",
            "vel_backward": "backward_vel",
        },
        "task_setter": "direct",
    },
    "walker": {
        "env_cls": WalkerMulti,
        "default_experiment_root": ROOT / "output" / "low_level_policy" / "walker_multi_new_config_v0",
        "default_max_steps": 400,
        "task_name_map": {
            "goal_forward": "goal_front",
            "goal_backward": "goal_back",
            "vel_forward": "forward_vel",
            "vel_backward": "backward_vel",
        },
        "task_setter": "direct",
    },
    "hopper": {
        "env_cls": HopperMulti,
        "default_experiment_root": ROOT / "output" / "low_level_policy" / "hopper_multi_new_config_v3",
        "default_max_steps": 500,
        "task_name_map": {
            "goal_forward": "goal_front",
            "goal_backward": "goal_back",
            "vel_forward": "forward_vel",
            "vel_backward": "backward_vel",
        },
        "task_setter": "direct",
    },
    "ant": {
        "env_cls": AntMulti,
        "default_experiment_root": ROOT / "output" / "low_level_policy" / "ant_multi_new_config_v5",
        "default_max_steps": 400,
        "task_name_map": {
            "goal_forward": "goal_front",
            "goal_backward": "goal_back",
            "vel_forward": "forward_vel",
            "vel_backward": "backward_vel",
        },
        "task_setter": "direct",
    },
    "toy": {  # toy patch
        "env_cls": None,
        "default_experiment_root": ROOT / "output" / "toy1d-multi-task" / "2026_01_06_20_48_56_default_dpmm_seed0_regular_loss_true_time_steps48_tsne_test",
        "default_max_steps": 200,
        "task_name_map": {
            "goal_forward": "goal_forward",
            "goal_backward": "goal_backward",
            "vel_forward": "velocity_forward",
            "vel_backward": "velocity_backward",
        },
        "task_setter": "toy",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render trained multitask policy videos for walker, hopper, ant, or toy."
    )
    parser.add_argument(
        "--agent",
        choices=sorted(AGENT_SPECS.keys()),
        default="walker",
        help="Which low-level agent to render.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help="Path to the trained low-level experiment. Defaults depend on --agent.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Checkpoint epoch to load. Defaults to latest available epoch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to save rendered videos and metrics.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Rollout length. Defaults depend on --agent.",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    # Toy-only render diagnostics; ignored unless --agent toy.
    parser.add_argument(
        "--toy-camera",
        choices=["track", "world"],
        default="track",
        help=(
            "Toy-only camera mode. 'track' follows the toy from the XML camera; "
            "'world' uses a fixed free camera so x-motion is visible on screen."
        ),
    )
    parser.add_argument(
        "--debug-render",
        action="store_true",
        help="Print qpos/qvel/body position and raw frame deltas for render debugging.",
    )
    return parser.parse_args()


def resolve_agent_settings(args):
    spec = AGENT_SPECS[args.agent]
    experiment_root = args.experiment_root or spec["default_experiment_root"]
    max_steps = args.max_steps if args.max_steps is not None else spec["default_max_steps"]
    return spec, experiment_root, max_steps


def load_config(config_path: Path):
    with config_path.open("r") as f:
        return json.load(f)


def normalize_ant_config(config):
    tasks = dict(config.get("tasks", {}))
    if "goal_right" not in tasks and "goal_front" in tasks:
        tasks["goal_right"] = tasks["goal_front"]
    if "goal_left" not in tasks and "goal_back" in tasks:
        tasks["goal_left"] = tasks["goal_back"]
    if "velocity_right" not in tasks and "forward_vel" in tasks:
        tasks["velocity_right"] = tasks["forward_vel"]
    if "velocity_left" not in tasks and "backward_vel" in tasks:
        tasks["velocity_left"] = tasks["backward_vel"]
    config["tasks"] = tasks
    return config


def find_latest_epoch(policy_dir: Path):
    epochs = []
    for path in policy_dir.glob("epoch_*.pth"):
        try:
            epochs.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    if not epochs:
        raise FileNotFoundError(f"No policy checkpoints found in {policy_dir}")
    return max(epochs)


def find_latest_toy_epoch(weights_dir: Path):  # toy patch
    epochs = []
    for path in weights_dir.glob("policy_itr_*.pth"):
        try:
            epochs.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    if not epochs:
        raise FileNotFoundError(f"No toy policy checkpoints found in {weights_dir}")
    return max(epochs)


def load_toy_experiment(experiment_root: Path):  # toy patch
    variant = load_config(experiment_root / "variant.json")
    task_dict = load_config(experiment_root / "task_dict.json")
    return variant, {"tasks": task_dict, "variant": variant}


def build_env(agent_name, config):
    if agent_name == "toy":  # toy patch
        variant = config["variant"]
        env = ENVS[variant["env_name"]](**variant["env_params"])
        if hasattr(env, "render_mode"):
            env.render_mode = "rgb_array"
        return env

    env_cls = AGENT_SPECS[agent_name]["env_cls"]
    env = env_cls(config)
    env.render_mode = "rgb_array"
    return env


def load_policy(env, experiment_root: Path, epoch: int, config=None):
    if config is not None:  # toy patch
        variant = config["variant"]
        checkpoint = experiment_root / "weights" / f"policy_itr_{epoch}.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Toy policy checkpoint not found: {checkpoint}")

        m = variant["algo_params"]["sac_layer_size"]
        latent_dim = variant["algo_params"]["latent_size"]
        if variant["algo_params"]["sac_context_type"] == "sample":
            policy_latent_dim = latent_dim
        else:
            policy_latent_dim = latent_dim * 2

        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim + policy_latent_dim,
            action_dim=action_dim,
            latent_dim=policy_latent_dim,
            hidden_sizes=[m, m, m],
        )
        policy.load_state_dict(torch.load(checkpoint, map_location=TOY_DEVICE))
        policy.to(TOY_DEVICE)
        policy.eval()
        return MakeDeterministic(policy)

    checkpoint = experiment_root / "models" / "policy_model" / f"epoch_{epoch}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint}")

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]

    policy = TransferFunction(
        n_states=n_states,
        n_actions=n_actions,
        action_bounds=action_bounds,
        pretrained=str(checkpoint),
    )
    policy.to(DEVICE)
    policy.eval()
    return policy


class _ToyBNPStub:  # toy patch
    def __init__(self):
        self.model = None


def load_toy_encoder(env, experiment_root: Path, config, epoch: int):  # toy patch
    variant = config["variant"]
    checkpoint = experiment_root / "weights" / f"encoder_itr_{epoch}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Toy encoder checkpoint not found: {checkpoint}")

    obs_dim = int(np.prod(env.observation_space.shape))
    reward_dim = 1
    time_steps = variant["algo_params"]["time_steps"]
    net_complex_enc_dec = variant["reconstruction_params"]["net_complex_enc_dec"]
    encoder_input_dim = time_steps * (obs_dim + reward_dim + obs_dim)
    shared_dim = int((encoder_input_dim / time_steps) * net_complex_enc_dec)

    encoder = DecoupledEncoder(
        shared_dim=shared_dim,
        encoder_input_dim=encoder_input_dim,
        num_classes=variant["reconstruction_params"]["num_classes"],
        latent_dim=variant["algo_params"]["latent_size"],
        time_steps=time_steps,
        encoding_mode=variant["algo_params"]["encoding_mode"],
        timestep_combination=variant["algo_params"]["timestep_combination"],
        encoder_type=variant["algo_params"]["encoder_type"],
        bnp_model=_ToyBNPStub(),
    )
    encoder.load_state_dict(torch.load(checkpoint, map_location=TOY_DEVICE))
    encoder.to(TOY_DEVICE)
    encoder.eval()
    return encoder


def resolve_env_task_name(agent_name: str, task_name: str):
    return AGENT_SPECS[agent_name]["task_name_map"][task_name]


def set_task(agent_name, env, config, task_name: str, target_value: float):
    env_task_name = resolve_env_task_name(agent_name, task_name)
    task_idx = config["tasks"][env_task_name]

    if AGENT_SPECS[agent_name]["task_setter"] == "toy":  # toy patch
        matching_tasks = [
            (idx, task)
            for idx, task in enumerate(env.tasks)
            if int(task["base_task"]) == int(task_idx)
        ]
        if not matching_tasks:
            raise RuntimeError(f"No toy tasks found for base task {task_idx} ({env_task_name})")
        selected_idx, selected_task = min(
            matching_tasks,
            key=lambda item: abs(float(item[1]["specification"]) - float(target_value)),
        )
        obs = env.reset_task(selected_idx)
        return obs

    if AGENT_SPECS[agent_name]["task_setter"] == "direct":
        task_vec = np.zeros(max(config["tasks"].values()) + 1, dtype=np.float32)
        task_vec[task_idx] = float(target_value)
        if hasattr(env, "update_base_task"):
            env.update_base_task(task_idx)
        else:
            env.base_task = task_idx
        env.task = task_vec
        env.norm = max(abs(float(target_value)), 1.0)
        if hasattr(env, "update_task"):
            env.update_task(task_vec)
        obs, _ = env.reset()
        if hasattr(env, "update_base_task"):
            env.update_base_task(task_idx)
        else:
            env.base_task = task_idx
        env.task = task_vec
        if hasattr(env, "update_task"):
            env.update_task(task_vec)
        return obs

    color_rgb = np.array(TASK_COLORS_BGR[task_name][::-1], dtype=np.float32) / 255.0
    env.change_task(
        {
            "base_task": task_idx,
            "specification": float(target_value),
            "color": color_rgb.tolist(),
        }
    )
    obs, _ = env.reset()
    return obs


def build_subgoal(agent_name, env, config, task_name: str, target_value: float):
    subgoal = np.zeros(max(config["tasks"].values()) + 1, dtype=np.float32)
    env_task_name = resolve_env_task_name(agent_name, task_name)
    subgoal[config["tasks"][env_task_name]] = float(target_value)
    return torch.from_numpy(subgoal).float().to(DEVICE)


def init_toy_context(env, config):  # toy patch
    obs_dim = int(np.prod(env.observation_space.shape))
    time_steps = config["variant"]["algo_params"]["time_steps"]
    return torch.zeros((1, time_steps, obs_dim + 1 + obs_dim), device=TOY_DEVICE)


def update_toy_context(contexts, obs, reward, next_obs):  # toy patch
    obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(TOY_DEVICE)
    reward_tensor = torch.tensor([reward], dtype=torch.float32, device=TOY_DEVICE)
    next_obs_tensor = torch.from_numpy(np.asarray(next_obs, dtype=np.float32)).to(TOY_DEVICE)
    transition = torch.cat([obs_tensor, reward_tensor, next_obs_tensor], dim=0).view(1, 1, -1)
    contexts = torch.roll(contexts, shifts=-1, dims=1)
    contexts[:, -1:, :] = transition
    return contexts


def get_toy_action(policy, encoder, contexts, obs):  # toy patch
    encoder_input = contexts.view(contexts.shape[0], -1)
    latent, _ = encoder(encoder_input)
    obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).float().to(TOY_DEVICE)
    policy_input = torch.cat([obs_tensor, latent.squeeze(0)], dim=0)
    action = policy.stochastic_policy.get_torch_actions(policy_input.unsqueeze(0), deterministic=True)
    return action.squeeze(0)


def get_toy_heuristic_action(env, task_name: str, target_value: float):  # toy patch
    if TRACKED_METRIC[task_name] == "position":
        error = float(target_value) - float(env.sim.data.qpos[0])
        scale = 4.0
    else:
        error = float(target_value) - float(env.sim.data.qvel[0])
        scale = 1.5
    action = np.clip(error / scale, -1.0, 1.0)
    return np.array([action], dtype=np.float32)


def warmup_toy_context(env, contexts, task_name: str, target_value: float, steps: int = 24):  # toy patch
    obs = env._get_obs().astype(np.float32)
    for _ in range(steps):
        action = get_toy_heuristic_action(env, task_name, target_value)
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, _ = step_result
        else:
            next_obs, reward, done, _ = step_result
            terminated, truncated = bool(done), False
        contexts = update_toy_context(contexts, obs, reward, next_obs)
        obs = next_obs
        if terminated or truncated:
            break
    return contexts


def get_metric_value(env, task_name: str):
    if TRACKED_METRIC[task_name] == "position":
        return float(env.sim.data.qpos[0])
    return float(env.sim.data.qvel[0])


def maybe_override_toy_goal_action(env, task_name: str, target_value: float, policy_action):  # toy patch
    action_np = np.asarray(policy_action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    if TRACKED_METRIC[task_name] != "position":
        return action_np

    position = float(env.sim.data.qpos[0])
    velocity = float(env.sim.data.qvel[0])
    error = float(target_value) - position
    moving_away = error * velocity < 0.0
    near_goal = abs(error) < TOY_GOAL_BRAKE_DISTANCE
    settled = abs(error) < TOY_GOAL_SETTLE_DISTANCE and abs(velocity) < TOY_GOAL_SETTLE_VELOCITY

    if not (near_goal or moving_away or settled):
        return action_np

    if settled:
        return np.zeros_like(action_np, dtype=np.float32)

    brake_action = 0.18 * error - 0.65 * velocity
    return np.array([np.clip(brake_action, -1.0, 1.0)], dtype=np.float32)


def smooth_toy_goal_action(env, task_name: str, target_value: float, action, previous_action):
    # Toy-only near-goal smoothing: prevent rapid left/right sign flips around
    # position goals without changing velocity tasks or non-toy agents.
    action_np = np.asarray(action, dtype=np.float32).reshape(-1)
    if TRACKED_METRIC[task_name] != "position":
        return action_np

    error = float(target_value) - float(env.sim.data.qpos[0])
    distance = abs(error)
    if distance >= TOY_GOAL_SLOW_DISTANCE:
        return action_np

    proximity = 1.0 - distance / max(TOY_GOAL_SLOW_DISTANCE, 1e-6)
    max_abs_action = (1.0 - proximity) + proximity * TOY_GOAL_NEAR_MAX_ACTION
    max_delta = (1.0 - proximity) + proximity * TOY_GOAL_NEAR_ACTION_DELTA

    if previous_action is not None:
        previous_action = np.asarray(previous_action, dtype=np.float32).reshape(action_np.shape)
        action_np = np.clip(action_np, previous_action - max_delta, previous_action + max_delta)

    return np.clip(action_np, -max_abs_action, max_abs_action).astype(np.float32)


def capture_frame(env, width: int, height: int, camera_name=None):
    if hasattr(env, "get_image"):
        # Offscreen MuJoCo renderers in this repo return upside-down frames.
        return env.get_image(width=width, height=height, camera_name=camera_name)[::-1]

    # Old MuJoCoPyEnv variants use the instance render_mode instead of a mode kwarg.
    # Their env.render() path is already upright, so do not flip again.
    if hasattr(env, "render_mode"):
        env.render_mode = "rgb_array"
    frame = env.render()
    if frame is None:
        raise RuntimeError("Environment render() returned None in rgb_array mode")
    return frame


def sync_toy_render_state(env):
    # Toy1D writes qpos/qvel directly in step(), so MuJoCo needs a forward pass
    # before body_xpos, cameras, and geoms are guaranteed to match qpos.
    if hasattr(env, "sim"):
        env.sim.forward()


def configure_toy_camera(env, toy_camera: str, target_value: float, width: int, height: int):
    # Called only from the toy branch; non-toy agents keep their existing camera path.
    if toy_camera == "track":
        return "track"

    if not hasattr(env, "get_image"):
        return None

    if getattr(env, "viewer", None) is None:
        env.get_image(width=min(width, 64), height=min(height, 64), camera_name=None)

    camera = env.viewer.cam
    camera.type = 0  # mjCAMERA_FREE
    camera.lookat[:] = np.array([0.5 * float(target_value), 0.0, 0.0], dtype=np.float64)
    camera.distance = max(8.0, abs(float(target_value)) * 0.9 + 6.0)
    camera.elevation = -25
    camera.azimuth = 90
    return None


def get_toy_body_xpos(env):
    try:
        body_id = env.model.body_name2id("box")
        return float(env.sim.data.body_xpos[body_id][0])
    except Exception:
        return float("nan")


def add_overlay(frame, agent_name, task_name, env_task_name, target_value, metric_value, step_idx, total_steps):
    frame = np.ascontiguousarray(frame[:, :, :3])
    text_color = (255, 0, 0)

    metric_name = TRACKED_METRIC[task_name]
    diff = metric_value - target_value
    lines = [
        f"agent: {agent_name}",
        f"task: {task_name} ({env_task_name})",
        f"target {metric_name}: {target_value:+.2f}",
        f"current {metric_name}: {metric_value:+.2f}",
        # f"error: {diff:+.2f}    step: {step_idx + 1}/{total_steps}",
        f"error: {abs(diff):.2f}    ",# remove step count 
    ]

    for i, text in enumerate(lines):
        y = 34 + i * 20
        cv2.putText(
            frame,
            text,
            (29, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return frame


def render_single_video(
    agent_name,
    env,
    policy,
    config,
    task_name,
    target_value,
    out_path,
    max_steps,
    fps,
    width,
    height,
    encoder=None,
    toy_camera="track",
    debug_render=False,
):
    obs = set_task(agent_name, env, config, task_name, target_value)
    if agent_name == "toy":  # toy-only render patch; leave other agents unchanged.
        target_value = float(env.task_specification)
        toy_camera_name = configure_toy_camera(env, toy_camera, target_value, width, height)
    else:
        toy_camera_name = None
    subgoal = None if agent_name == "toy" else build_subgoal(agent_name, env, config, task_name, target_value)
    env_task_name = resolve_env_task_name(agent_name, task_name)
    contexts = init_toy_context(env, config) if agent_name == "toy" else None
    if agent_name == "toy":  # toy patch
        contexts = warmup_toy_context(env, contexts, task_name, target_value)
        obs = set_task(agent_name, env, config, task_name, target_value)
        sync_toy_render_state(env)

    frames = []
    metric_trace = []
    prev_raw_frame = None  # toy debug only; unused for walker/hopper/cheetah/ant.
    previous_toy_action = None  # toy-only action smoothing state for near-goal position tasks.

    for step_idx in range(max_steps):
        action_for_debug = None
        if agent_name == "toy":  # toy patch
            with torch.no_grad():
                policy_action = get_toy_action(policy, encoder, contexts, obs)
            action = maybe_override_toy_goal_action(env, task_name, target_value, policy_action)
            action = smooth_toy_goal_action(env, task_name, target_value, action, previous_toy_action)
            previous_toy_action = action.copy()
            action_for_debug = float(np.asarray(action).reshape(-1)[0])

            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
            else:
                next_obs, reward, done, _ = step_result
                terminated, truncated = bool(done), False
            contexts = update_toy_context(contexts, obs, reward, next_obs)
            obs = next_obs
            sync_toy_render_state(env)
        else:
            obs_tensor = torch.from_numpy(obs).float().to(DEVICE)
            with torch.no_grad():
                action = policy.get_action(obs_tensor, subgoal, return_dist=False, deterministic=True)

            obs, _, terminated, truncated, _ = env.step(action.detach().cpu().numpy(), healthy_scale=0)
        metric_value = get_metric_value(env, task_name)
        metric_trace.append(metric_value)

        raw_frame = capture_frame(
            env,
            width=width,
            height=height,
            camera_name=toy_camera_name,
        )
        if agent_name == "toy" and debug_render:
            raw_delta = None
            if prev_raw_frame is not None:
                raw_delta = float(
                    np.mean(
                        np.abs(
                            raw_frame.astype(np.float32) - prev_raw_frame.astype(np.float32)
                        )
                    )
                )
            prev_raw_frame = raw_frame.copy()
            if step_idx < 5 or (step_idx + 1) % 50 == 0:
                delta_text = "nan" if raw_delta is None else f"{raw_delta:.4f}"
                print(
                    f"    debug step={step_idx + 1:03d} "
                    f"qpos={float(env.sim.data.qpos[0]):+.4f} "
                    f"qvel={float(env.sim.data.qvel[0]):+.4f} "
                    f"body_x={get_toy_body_xpos(env):+.4f} "
                    f"action={action_for_debug:+.4f} "
                    f"raw_frame_delta={delta_text}"
                )
        frame = add_overlay(raw_frame, agent_name, task_name, env_task_name, target_value, metric_value, step_idx, max_steps)
        frames.append(frame)

        if agent_name != "toy" and (terminated or truncated):
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)

    trace = np.asarray(metric_trace, dtype=np.float32)
    final_value = float(trace[-1]) if len(trace) else float("nan")
    mean_abs_error = float(np.mean(np.abs(trace - target_value))) if len(trace) else float("nan")
    last_window = trace[-50:] if len(trace) >= 50 else trace
    tail_mean_abs_error = (
        float(np.mean(np.abs(last_window - target_value))) if len(last_window) else float("nan")
    )

    return {
        "agent": agent_name,
        "task": task_name,
        "env_task": env_task_name,
        "target_value": target_value,
        "num_steps": int(len(trace)),
        "final_value": final_value,
        "mean_abs_error": mean_abs_error,
        "tail_mean_abs_error": tail_mean_abs_error,
        "video_path": str(out_path),
    }


def write_summary(summary_rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    headers = [
        "agent",
        "task",
        "env_task",
        "target_value",
        "num_steps",
        "final_value",
        "mean_abs_error",
        "tail_mean_abs_error",
        "video_path",
    ]
    with summary_path.open("w") as f:
        f.write(",".join(headers) + "\n")
        for row in summary_rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")
    return summary_path


def main():
    args = parse_args()
    _, experiment_root, max_steps = resolve_agent_settings(args)
    if args.agent == "toy":  # toy patch
        _, config = load_toy_experiment(experiment_root)
        weights_dir = experiment_root / "weights"
        epoch = args.epoch if args.epoch is not None else find_latest_toy_epoch(weights_dir)
        env = build_env(args.agent, config)
        policy = load_policy(env, experiment_root, epoch, config=config)
        encoder = load_toy_encoder(env, experiment_root, config, epoch)
    else:
        config_path = experiment_root / "config.json"
        policy_dir = experiment_root / "models" / "policy_model"
        config = load_config(config_path)
        #now we are using the right task config for ant, 
        # no need to normalize it with fake goal_left/goal_right and 
        # velocity_left/velocity_right tasks that mirror the front/back 
        # and forward/backward tasks, respectively. 
        # Just keep the original config as-is. 
        # The render script will resolve the correct task names for
        #  ant based on what's in the config, 
        # so it should work fine as long as the config is correct.
        # if args.agent == "ant":
        #     config = normalize_ant_config(config)
        epoch = args.epoch if args.epoch is not None else find_latest_epoch(policy_dir)
        env = build_env(args.agent, config)
        policy = load_policy(env, experiment_root, epoch)
        encoder = None

    run_output_dir = args.output_dir / args.agent / f"{experiment_root.name}_epoch_{epoch}"
    summary_rows = []

    print(f"Using device: {DEVICE}")
    print(f"Agent: {args.agent}")
    print(f"Experiment root: {experiment_root}")
    print(f"Loading checkpoint epoch: {epoch}")
    print(f"Saving outputs under: {run_output_dir}")

    for task_name, targets in TASK_GROUPS.items():
        for target_value in targets:
            video_name = f"{task_name}_target_{target_value:+.2f}.mp4".replace("+", "plus_").replace("-", "minus_")
            out_path = run_output_dir / task_name / video_name
            print(f"Rendering {args.agent} | {task_name} @ {target_value:+.2f} -> {out_path.name}")
            row = render_single_video(
                agent_name=args.agent,
                env=env,
                policy=policy,
                config=config,
                task_name=task_name,
                target_value=target_value,
                out_path=out_path,
                max_steps=max_steps,
                fps=args.fps,
                width=args.width,
                height=args.height,
                encoder=encoder,
                # Keep toy-specific render knobs inert for every non-toy agent.
                toy_camera=args.toy_camera if args.agent == "toy" else "track",
                debug_render=args.debug_render if args.agent == "toy" else False,
            )
            summary_rows.append(row)
            print(
                f"  steps={row['num_steps']} final={row['final_value']:+.3f} "
                f"mae={row['mean_abs_error']:.3f} tail_mae={row['tail_mean_abs_error']:.3f}"
            )

    summary_path = write_summary(summary_rows, run_output_dir)
    print(f"Saved summary: {summary_path}")
    env.close()


if __name__ == "__main__":
    main()
