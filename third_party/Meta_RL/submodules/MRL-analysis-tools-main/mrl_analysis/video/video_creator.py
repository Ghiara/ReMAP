# import sys
# import numpy as np
# import cv2
# from collections import deque
# from typing import List, Any, Dict, Callable
# from gym.envs.mujoco.mujoco_env import MujocoEnv, BaseMujocoEnv
# from pathlib import Path

# from mrl_analysis.utility import interfaces
# from .camera_wrapper import CameraWrapper


# class VideoCreator():
#     """A class for video rendering of policies.

#     Use function ``create_video`` to render environment interactions and write
#     the results to a video.

#     You can modify ``fps`` to set the number of rendered frames per second.
#     """
#     def __init__(self) -> None:
#         self.fps = 10.0
#         self.buffer_size = 10

#         # Context arrays
#         self.observations: deque
#         self.actions: deque
#         self.rewards: deque
#         self.next_observations: deque
#         self.terminals: deque
#         self.environment_steps = 0
#         self.obs: np.ndarray = None

#     def _init_buffers(self, encoder: interfaces.MdpEncoder, decoder: interfaces.MdpDecoder):
#         """ Initialize buffers (used for context accumulation) """
#         # Context data initialization
#         self.observations = deque(
#             [np.zeros((decoder.observation_dim)) for _ in range(encoder.context_size)],
#             maxlen=encoder.context_size
#         )
#         self.actions = deque(
#             [np.zeros((decoder.action_dim)) for _ in range(encoder.context_size)],
#             maxlen=encoder.context_size
#         )
#         self.rewards = deque(
#             [np.zeros((1)) for _ in range(encoder.context_size)],
#             maxlen=encoder.context_size
#         )
#         self.next_observations = deque(
#             [np.zeros((decoder.observation_dim)) for _ in range(encoder.context_size)],
#             maxlen=encoder.context_size
#         )
#         self.terminals = deque(
#             [np.zeros((1)) for _ in range(encoder.context_size)],
#             maxlen=encoder.context_size
#         )
#         self.obs = None

#     def create_video(
#         self,
#         results: Dict[str, Any],
#         video_length: float = None,
#         n_frames: int = None,
#         save_as: str = "./videos/video.mp4",
#         env_reset_interval: int = 250,
#         width: int = None,
#         height: int = None,
#     ) -> None:
#         """Create a video file from an environment, an encoder, and a policy

#         Parameters
#         ----------
#         results : Dict[str, Any]
#             Dictionary of (training) results, must include the following items:
#                 eval_env: ``interfaces.MetaEnv`` - Environment
#                 encoder: ``interfaces.MdpEncoder`` - Context encoder
#                 decoder: ``interfaces.MdpDecoder`` - Decoder (maps context + state + action to predictions)
#                 policy: ``interfaces.MetaRLPolicy`` - Policy for interaction, takes state + encoding as inputs
#         video_length : float
#             Length of the video in frames
#         n_frames : int
#             Number of frames in the video, can be provided *instead of video_length*
#         save_as : str, optional
#             Name of video file, by default "./videos/video.mp4"
#         env_reset_interval : int, optional
#             Reset interval for environment (steps), by default 250
#         width : int, optional
#             Width of rendered video (in pixels), by default None
#         height : int, optional
#             Height of rendered video (in pixels), by default None
#         """

#         if video_length is not None:
#             n_frames = int(self.fps * video_length)
#         if video_length is None:
#             raise ValueError("You need to provide either 'video_length' of 'n_frames'.")

#         # Create output path if it doesn't already exist
#         Path(save_as).parent.mkdir(parents=True, exist_ok=True)

#         # Read variables from results
#         env: interfaces.MetaEnv = results['eval_env']
#         encoder: interfaces.MdpEncoder = results['encoder']
#         decoder: interfaces.MdpDecoder = results['decoder']
#         policy: interfaces.MetaRLPolicy = results['policy']
#         encoder.train(False)
#         decoder.train(False)
#         policy.train(False)

#         # Transformation of each frame before it is written to video
#         transform = lambda x: x

#         # MujocoEnv support
#         if isinstance(env.unwrapped, MujocoEnv) or isinstance(env.unwrapped, BaseMujocoEnv):
#             env = CameraWrapper(env)
#             transform = np.flipud

#         self._init_buffers(encoder, decoder)
#         collected_frames = 0
#         self.environment_steps = 0

#         # Collect frames and write them to the video file
#         video: cv2.VideoWriter = None
#         while collected_frames < n_frames:
#             # Collect frames by simulation
#             frames = self._collect_frames(
#                 env, encoder, policy, 
#                 min(n_frames - collected_frames, self.buffer_size), 
#                 env_reset_interval=env_reset_interval, w=width, h=height
#             )
#             collected_frames += len(frames)

#             # Apply frame transformations
#             frames = [transform(frame) for frame in frames]

#             # Initialize VideoWriter (only once)
#             if video is None:
#                 size = frames[0].shape
#                 video = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (size[1], size[0]), True)
            
#             # Write frames to video
#             self._frames_to_video(video, frames, transform)
#             sys.stdout.write(f"\r{collected_frames} of {n_frames} frames collected.")
#         video.release()
#         print("\nDone!")

#     def _collect_frames(
#         self,
#         env: interfaces.MetaEnv,
#         encoder: interfaces.MdpEncoder,
#         policy: interfaces.MetaRLPolicy,
#         n_frames: int,
#         env_reset_interval: int = 250,
#         w: int = None,  # width
#         h: int = None,  # height
#     ) -> List[np.ndarray]:
#         """ Renders environment interactions and returns the frames """

#         # Frame sampling by policy rollout
#         frames = []
#         if self.obs is None:
#             self.obs, _ = env.reset()
#         for t in range(n_frames):
#             context = {
#                 'observations': np.array(self.observations),
#                 'actions': np.array(self.actions),
#                 'rewards': np.array(self.rewards),
#                 'next_observations': np.array(self.next_observations),
#                 'terminals': np.array(self.terminals)
#             }
#             latent = encoder.get_encoding(**context)
            
#             action, _ = policy.get_action(self.obs, latent)
#             next_obs, reward, terminal, _, info = env.step(action)

#             kwargs = {}
#             if w is not None: kwargs['width'] = w
#             if h is not None: kwargs['height'] = h
#             image = env.render('rgb_array', **kwargs)
#             frames.append(image)

#             self.observations.append(self.obs)
#             self.actions.append(action)
#             self.rewards.append(np.array([reward]))
#             self.next_observations.append(next_obs)
#             self.terminals.append(np.array([terminal]))
#             self.obs = next_obs
#             self.environment_steps += 1

#             if self.environment_steps % env_reset_interval == 0 or terminal:
#                 env.sample_task()
#                 self.obs, _ = env.reset()

#         return frames
    
#     def _frames_to_video(self, video: cv2.VideoWriter, frames: List[np.ndarray], transform: Callable = None):
#         """ Write collected frames to video file """
#         for frame in frames:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
#             video.write(frame)



# # --------------------------------------------------FOR OLD REFERENCE ONLY--------------------------------------------------

#-----------------------------
#FIrst version of

# import sys
# import numpy as np
# import cv2
# from collections import deque
# from typing import List, Any, Dict, Callable
# from pathlib import Path
# from gym.envs.mujoco.mujoco_env import MujocoEnv, BaseMujocoEnv

# from mrl_analysis.utility import interfaces
# import torch



# class VideoCreator():
#     """A class for Mujoco video rendering of policies."""

#     def __init__(self) -> None:
#         self.fps = 30.0
#         self.buffer_size = 10
#         self.environment_steps = 0
#         self.obs: np.ndarray = None

#     # --------------------------------------------------------------------------
#     # Optional: buffer initialization (used by encoder)
#     # --------------------------------------------------------------------------
#     def _init_buffers(self, encoder: interfaces.MdpEncoder, decoder: interfaces.MdpDecoder):
#         """Initialize buffers (used for context accumulation)"""
#         # Ensure decoder.observation_dim and action_dim are numeric
#         obs_dim = decoder.observation_dim
#         act_dim = decoder.action_dim
#         if not isinstance(obs_dim, int):
#             try:
#                 obs_dim = int(np.prod(obs_dim.shape))
#             except AttributeError:
#                 # if it's a gym.Box, extract shape[0]
#                 obs_dim = int(np.prod(obs_dim.shape)) if hasattr(obs_dim, "shape") else 1
#         if not isinstance(act_dim, int):
#             try:
#                 act_dim = int(np.prod(act_dim.shape))
#             except AttributeError:
#                 act_dim = int(np.prod(act_dim.shape)) if hasattr(act_dim, "shape") else 1

#         # Initialize context buffers
#         self.observations = deque([np.zeros(obs_dim) for _ in range(encoder.context_size)], maxlen=encoder.context_size)
#         self.actions = deque([np.zeros(act_dim) for _ in range(encoder.context_size)], maxlen=encoder.context_size)
#         self.rewards = deque([np.zeros(1) for _ in range(encoder.context_size)], maxlen=encoder.context_size)
#         self.next_observations = deque([np.zeros(obs_dim) for _ in range(encoder.context_size)], maxlen=encoder.context_size)
#         self.terminals = deque([np.zeros(1) for _ in range(encoder.context_size)], maxlen=encoder.context_size)
#         self.obs = None

#     def create_video(
#         self,
#         results: Dict[str, Any],
#         video_length: float = 10.0,
#         save_as: str = "./videos/video.mp4",
#         env_reset_interval: int = 250,
#         width: int = 1280,
#         height: int = 720,
#         override_env: Any = None,
#     ) -> None:
#         """Render Mujoco rollout video from encoder + low-level policy."""
#         Path(save_as).parent.mkdir(parents=True, exist_ok=True)

#         encoder: interfaces.MdpEncoder = results["encoder"]
#         decoder: interfaces.MdpDecoder = results["decoder"]
#         policy: interfaces.MetaRLPolicy = results.get("eval_policy", results.get("policy"))

#         # choose env
#         env = override_env if override_env is not None else results["eval_env"]
#         if isinstance(env.unwrapped, (MujocoEnv, BaseMujocoEnv)):
#             env.render_mode = "rgb_array"

#         # ---------- helper: robust goal extraction ----------
#         def _extract_goal(env) -> Any:
#             """Try to robustly extract current goal value from AntMulti or other mujoco envs."""
#             goal_val = None

#             # --- 0. if env.task is numpy array (common in AntMulti) ---
#             if hasattr(env, "task") and isinstance(env.task, (np.ndarray, list)):
#                 goal_val = env.task
#             # --- 1. try direct attributes ---
#             elif hasattr(env, "goal_position"):
#                 goal_val = getattr(env, "goal_position")
#             elif hasattr(env, "goal"):
#                 goal_val = getattr(env, "goal")
#             elif hasattr(env, "current_goal"):
#                 goal_val = getattr(env, "current_goal")
#             # --- 2. check inside task dict ---
#             elif hasattr(env, "task") and isinstance(env.task, dict):
#                 for k in ("goal_right", "goal_left", "goal", "target", "x_goal", "x_target"):
#                     if k in env.task:
#                         goal_val = env.task[k]
#                         break

#             # --- convert to displayable string ---
#             if goal_val is None:
#                 return "N/A"
#             else:
#                 goal_arr = np.asarray(goal_val).reshape(-1)
#                 return f"{goal_arr[0]:.2f}"


#         # ---------- helper: safe render ----------
#         def safe_render(env, width=1280, height=720):
#             frame = None
#             try:
#                 frame = env.render()
#             except TypeError:
#                 try:
#                     frame = env.render('rgb_array', width=width, height=height)
#                 except Exception:
#                     frame = None
#             if frame is not None:
#                 frame = np.ascontiguousarray(frame)
#                 if frame.dtype != np.uint8:
#                     frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
#                 if frame.ndim == 2:
#                     frame = np.repeat(frame[..., None], 3, axis=-1)
#                 if frame.shape[-1] == 4:
#                     frame = frame[..., :3]
#             return frame

#         # init modules
#         encoder.train(False); decoder.train(False); policy.train(False)
#         self._init_buffers(encoder, decoder)

#         # ✅ 先采样一次任务，再 reset
#         if hasattr(env, "sample_task"):
#             try: env.sample_task()
#             except Exception: pass
#         self.obs, _ = env.reset()

#         # 估计 policy 输入总维度（obs+latent）
#         try:
#             expected_input_dim = next(policy.parameters()).shape[1]
#             latent_dim = 1
#         except Exception:
#             expected_input_dim = None
#             latent_dim = 1

#         n_frames = int(self.fps * video_length)
#         video = None
#         self.environment_steps = 0
#         print(f"Recording {n_frames} frames to {save_as}")

#         for i in range(n_frames):
#             # context -> latent
#             context = {
#                 "observations": np.array(self.observations),
#                 "actions": np.array(self.actions),
#                 "rewards": np.array(self.rewards),
#                 "next_observations": np.array(self.next_observations),
#                 "terminals": np.array(self.terminals),
#             }
#             latent = encoder.get_encoding(**context)
#             latent = np.array(latent).flatten()
#             if latent.ndim == 0:
#                 latent = np.array([latent])

#             # obs -> tensor (+ auto pad/trim)
#             obs_np = np.array(self.obs, dtype=np.float32).flatten()
#             if expected_input_dim is not None:
#                 total = obs_np.shape[0] + latent.shape[0]
#                 if total < expected_input_dim:
#                     obs_np = np.concatenate([obs_np, np.zeros(expected_input_dim - total, dtype=np.float32)])
#                 elif total > expected_input_dim:
#                     obs_np = obs_np[:(obs_np.shape[0] - (total - expected_input_dim))]
#             obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
#             latent_tensor = torch.tensor(latent, dtype=torch.float32).unsqueeze(0)

#             # policy forward
#             with torch.no_grad():
#                 action_out = policy.get_action(obs_tensor, latent_tensor)
#             action = action_out[0] if isinstance(action_out, (tuple, list)) else action_out
#             action = action.detach().cpu().numpy().squeeze()

#             # env step
#             next_obs, reward, done, _, _ = env.step(action)

#             # render
#             frame = safe_render(env, width, height)
#             if frame is None:
#                 print(f"[WARN] Frame {i} render returned None — skipping frame.")
#                 continue

#             # overlay
#             try:
#                 x_pos = float(env.sim.data.qpos[0])
#                 goal_val = _extract_goal(env)
#                 if goal_val is None:
#                     goal_str = "N/A"
#                 else:
#                     arr = np.asarray(goal_val).reshape(-1)
#                     goal_str = f"{float(arr[0]):.2f}"

#                 text_lines = [
#                     f"Step: {i+1}/{n_frames}",
#                     f"X_pos: {x_pos:7.3f}",
#                     f"Goal: {goal_str}",
#                     f"Reward: {reward:7.3f}",
#                 ]
#                 y0, dy, color = 35, 28, (255, 255, 255)
#                 for j, t in enumerate(text_lines):
#                     cv2.putText(frame, t, (20, y0 + j * dy),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
#             except Exception as e:
#                 print(f"[WARN] Overlay failed at frame {i}: {e}")

#             # init writer & write
#             if video is None:
#                 size = (frame.shape[1], frame.shape[0])
#                 video = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
#             video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

#             # update buffers
#             self.observations.append(self.obs)
#             self.actions.append(action)
#             self.rewards.append(np.array([reward]))
#             self.next_observations.append(next_obs)
#             self.terminals.append(np.array([done]))
#             self.obs = next_obs
#             self.environment_steps += 1

#             # ✅ 每个周期/终止都先采样再 reset
#             if done or self.environment_steps % env_reset_interval == 0:
#                 if hasattr(env, "sample_task"):
#                     try: env.sample_task()
#                     except Exception: pass
#                 self.obs, _ = env.reset()

#             sys.stdout.write(f"\rFrame {i+1}/{n_frames}")
#             sys.stdout.flush()

#         if video is not None:
#             video.release()
#         print(f"\n Done! Saved Mujoco video: {save_as}")






#-----------------------------------------------------------
# second version of creating video in new_model_eval.py
# --------------------------------------------------------------------------



import os
import cv2
import numpy as np
import torch
import imageio
from tqdm import tqdm


class VideoCreator:
    """
    Enhanced video generator for Meta-RL or transfer-learning evaluations.
    Renders Mujoco environments with policy rollouts and overlays textual info.

    Features:
    - Works with Gym/Mujoco environments
    - Supports PyTorch policies (MetaRL style)
    - Displays goal, reward, and step counters on frames
    - Compatible with new_model_eval.py and result_dict structure
    """

    def __init__(self, fps=30, font_scale=0.6, font_color=(255, 255, 255), thickness=1):
        self.fps = fps
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_color = font_color
        self.thickness = thickness

    def create_video(
        self,
        result_dict,
        save_as,
        video_length=20.0,
        env_reset_interval=200,
        width=640,
        height=480,
    ):
        """
        Creates a rollout video with overlayed goal/reward/task info.

        Parameters
        ----------
        result_dict : dict
            Must include 'eval_env' and 'eval_policy'
        save_as : str
            Output .mp4 path
        video_length : float, optional
            Length of video in seconds, by default 20.0
        env_reset_interval : int, optional
            Number of steps before resetting, by default 200
        width : int
            Render width
        height : int
            Render height
        """

        # Prepare environment and policy
        env = result_dict.get("eval_env", None)
        policy = result_dict.get("eval_policy", None)
        if env is None or policy is None:
            raise ValueError("Result dict must contain 'eval_env' and 'eval_policy'.")

        from third_party.Meta_RL.smrl.policies.meta_policy import MakeDeterministic
        if isinstance(policy, MakeDeterministic):
            policy = policy.stochastic_policy

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)
        policy.eval()

        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        total_frames = int(self.fps * video_length)
        print(f"Generating {total_frames} frames at {self.fps} FPS...")

        video_writer = imageio.get_writer(save_as, fps=self.fps)
        episode = 0
        step = 0
        done = True
        obs = None

        # Rollout loop
        for frame_idx in tqdm(range(total_frames), desc="Rendering video"):
            # Reset periodically
            if done or step % env_reset_interval == 0 or obs is None:
                obs = env.reset()
                episode += 1
                step = 0
                done = False

            # Convert observation safely to tensor
            if isinstance(obs, tuple):
                obs = obs[0]
            if isinstance(obs, list) and isinstance(obs[0], np.ndarray):
                obs = np.concatenate([o[np.newaxis, ...] for o in obs], axis=0).squeeze()
            elif isinstance(obs, list):
                obs = np.array(obs)
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            if obs.size == 0:
                raise ValueError(f"Observation is empty or invalid: {type(obs)}, value={obs}")
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


            # --- FIX: Compute latent encoding for policies requiring it ---
            latent = None
            if 'encoder' in result_dict:
                encoder = result_dict['encoder']
                if hasattr(encoder, 'get_encoding'):
                    # Build dummy context to get encoding
                    dummy_context = {
                        'observations': np.zeros((encoder.context_size, encoder.observation_dim)),
                        'actions': np.zeros((encoder.context_size, encoder.action_dim)),
                        'rewards': np.zeros((encoder.context_size, 1)),
                        'next_observations': np.zeros((encoder.context_size, encoder.observation_dim)),
                        'terminals': np.zeros((encoder.context_size, 1))
                    }
                    latent = encoder.get_encoding(**dummy_context)
                    # Move latent to same device as obs_tensor
                    if isinstance(latent, np.ndarray):
                        latent = torch.tensor(latent, dtype=torch.float32, device=device)
                    elif isinstance(latent, torch.Tensor):
                        latent = latent.to(device)

            # --------------------------------------------------------------


            # Get action from policy
            with torch.no_grad():
                if hasattr(policy, "get_action"):
                    if latent is not None:
                        action, _ = policy.get_action(obs_tensor, latent)
                    else:
                        action, _ = policy.get_action(obs_tensor)

                elif hasattr(policy, "forward"):
                    action = policy(obs_tensor)
                else:
                    raise AttributeError("Policy has no get_action or forward method.")

            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).cpu().numpy()

            obs, reward, done, info = env.step(action)
            step += 1

            # Render frame
            try:
                frame = env.render(mode="rgb_array", width=width, height=height)
            except TypeError:
                frame = env.render(mode="rgb_array")
                frame = cv2.resize(frame, (width, height))

            # =============================
            # Overlay textual info
            # =============================
            overlay = frame.copy()

            # Try to extract goal/task info
            goal_str = ""
            if hasattr(env, "goal"):
                goal_str = str(np.round(env.goal, 3))
            elif isinstance(info, dict) and "goal" in info:
                goal_str = str(np.round(info["goal"], 3))

            # Compose display text
            lines = [
                f"Episode: {episode}",
                f"Step: {step}",
                f"Reward: {reward:.3f}",
                f"Goal: {goal_str}" if goal_str else "",
            ]

            y0 = 25
            for i, line in enumerate(lines):
                if line.strip() == "":
                    continue
                y = y0 + i * 20
                cv2.putText(
                    overlay,
                    line,
                    (10, y),
                    self.font,
                    self.font_scale,
                    self.font_color,
                    self.thickness,
                    cv2.LINE_AA,
                )

            # Blend overlay (optional: make semi-transparent)
            alpha = 0.9
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            video_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        video_writer.close()
        print(f"✅ Video saved at: {save_as}")



