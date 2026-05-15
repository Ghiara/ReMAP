Ran terminal command:  ls /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/run_task_inference_high_level_policy_training.py /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/train_low_level_policy.py 2>&1 && wc -l /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/run_task_inference_high_level_policy_training.py /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/train_low_level_policy.py

Ran terminal command:  cat /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/run_task_inference_high_level_policy_training.py

Ran terminal command:  cat /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/train_low_level_policy.py

Read [](file:///c%3A/Users/24498/AppData/Roaming/Code/User/workspaceStorage/b6490b41b18bca690657ce5e36a7c81e/GitHub.copilot-chat/chat-session-resources/38ea1435-b866-4f1b-b8b7-195a7f88fb90/toolu_bdrk_016UGQqD9Po7zYXqK6LAURNm__vscode-1778823266310/content.txt#1-1), lines 1 to 300

Read [](file:///c%3A/Users/24498/AppData/Roaming/Code/User/workspaceStorage/b6490b41b18bca690657ce5e36a7c81e/GitHub.copilot-chat/chat-session-resources/38ea1435-b866-4f1b-b8b7-195a7f88fb90/toolu_bdrk_01Jc9n4QW4Ng7zPWiPCs5xVN__vscode-1778823266311/content.txt#1-1), lines 1 to 334

Read [](file:///c%3A/Users/24498/AppData/Roaming/Code/User/workspaceStorage/b6490b41b18bca690657ce5e36a7c81e/GitHub.copilot-chat/chat-session-resources/38ea1435-b866-4f1b-b8b7-195a7f88fb90/toolu_bdrk_016UGQqD9Po7zYXqK6LAURNm__vscode-1778823266310/content.txt#300-300), lines 300 to 529

Now let me check a few key sub-modules to complete the picture: 

Ran terminal command:  grep -n "def collect_data\|def rollout\|def train\|replay_buffer\|\.add(\|\.store(" /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/tigr/rollout_worker.py 2>/dev/null | head -60

 

Ran terminal command:  grep -n "def train\|def _train\|reconstruction_trainer\|policy_trainer\|num_training_steps\|rollout_coordinator" /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/tigr/training_algorithm.py 2>/dev/null | head -60

Ran terminal command:  grep -n "def add_episode\|def sample\|def get_indices\|stats_dict" /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/tigr/stacked_replay_buffer.py 2>/dev/null | head -40

Ran terminal command:  grep -n "def train\|sample\|replay_buffer\|encoder\|decoder\|\.loss\|optimizer" /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/tigr/trainer/dpmm_trainer.py 2>/dev/null | head -60

Ran terminal command:  grep -n "def train\|def store\|replay_buffer\|memory\|sample\|encoder\|policy\|qf1\|qf2" /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/tigr/sac.py 2>/dev/null | head -50

Ran terminal command:  grep -n "def store\|def train\|def choose_action\|memory\|self\.memory\|replay" /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/agent.py 2>/dev/null | head -50

Ran terminal command:  grep -n "def store\|def train\|def choose_action\|memory\|self\.memory\|replay\|class SAC" /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/agent.py 2>/dev/null | head -50

Ran terminal command:  ls /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/agent.py && grep -n "def store\|def train\|def choose_action\|memory\|class SAC\|class " /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/agent.py | head -40

Ran terminal command:  find /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization -name "agent.py" 2>/dev/null && find /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization -name "*.py" | xargs grep -l "class SAC" 2>/dev/null | head -5

Ran terminal command:  grep -n "def store\|def train\|def choose_action\|class SAC\|memory\|Memory\|replay" /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/submodules/SAC/agent.py | head -40

Ran terminal command:  grep -n "def add\|class Memory\|class Transition\|save\|load\|checkpoint\|pth\|model" /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/submodules/SAC/replay_memory.py 2>/dev/null | head -30

Ran terminal command:  grep -n "save\|load\|torch.save\|log_all\|pth\|models/" /home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/vis_utils/vis_logging.py 2>/dev/null | head -30

Now I have all the information needed to write the comprehensive report.

---

# 系统架构详细报告：解耦高层-低层策略训练与部署

---

## 一、整体架构概览

系统由三个相互解耦的阶段组成：

```
┌─────────────────────────────┐
│  阶段①：高层训练              │
│  run_task_inference_...py   │
│  → 训练 Encoder + Decoder   │
│  → 训练高层 Policy (SAC)     │
└────────────┬────────────────┘
             │ 保存权重
             ▼
┌─────────────────────────────┐
│  阶段②：低层训练              │
│  train_low_level_policy.py  │
│  → 训练 TransferFunction    │
│    (PolicyNetwork, SAC)      │
└────────────┬────────────────┘
             │ 保存权重
             ▼
┌─────────────────────────────┐
│  阶段③：部署推理              │
│  task_inference_high_level  │
│  _cross_agent_deployment.py │
│  → 加载所有模块, 执行推理      │
│  → (可选) 在线 finetune      │
└─────────────────────────────┘
```

---

## 二、阶段①：高层训练 (run_task_inference_high_level_policy_training.py)

### 2.1 使用的环境

**简单环境**（`ENVS[variant['env_name']]`，如 toy 1D/2D 滑块环境）。观测维度低（如 obs_dim=2），动作空间简单（action_dim=1）。

### 2.2 采样流程

```
TrainingAlgorithm.train()
  └── RolloutCoordinator.collect_replay_data(tasks, num_transitions)
        └── RolloutCoordinator.rollout(n_tasks, deterministic=False)
              ├── agent.get_action(obs, context)   ← 调用 Encoder + Policy
              ├── env.step(action)                 ← 在简单环境中执行
              └── replay_buffer.add_episode(path, task_nr=task_id)
```

**每条 transition 的内容**：`(obs, action, reward, next_obs, task_id)`，按 episode 打包存入 `StackedReplayBuffer`。

### 2.3 数据保存位置

**全部在内存中**（`StackedReplayBuffer` 是纯内存结构，不写磁盘）。仅统计量 `stats_dict`（均值/方差，用于归一化）以及模型权重会写磁盘：

| 文件 | 路径 | 内容 |
|---|---|---|
| 各模块权重 | `{experiment_log_dir}/weights/encoder_itr_X.pth` | Encoder 权重 |
| | `{experiment_log_dir}/weights/policy_itr_X.pth` | Policy 权重 |
| | `{experiment_log_dir}/weights/decoder_itr_X.pth` | Decoder 权重 |
| | `{experiment_log_dir}/weights/qf1_itr_X.pth` 等 | Q网络权重 |
| 归一化统计 | `{experiment_log_dir}/weights/stats_dict.json` | obs/reward 归一化参数 |
| 配置 | `{experiment_log_dir}/variant.json` | 所有超参数 |
| 任务字典 | `{experiment_log_dir}/task_dict.json` | task名→编号映射 |

### 2.4 各模块的输入、输出与更新方式

#### 模块 A：`DecoupledEncoder`（任务推理编码器）

| 项目 | 内容 |
|---|---|
| **输入** | Context 片段 `(batch, time_steps, obs_dim + reward_dim + obs_dim)`，展平为 `(batch, time_steps*(2*obs_dim+1))` |
| **输出** | 潜变量均值 `μ (batch, latent_dim)` 和方差 `log_var (batch, latent_dim)` |
| **训练者** | `AugmentedTrainer.train(num_reconstruction_steps)` |
| **损失函数** | 重建损失（状态/奖励 MSE）+ 任务分类 CE 损失 + KL 散度（对齐 DPMM/GMM 混合先验）|
| **优化器** | `optimizer_mixture_model` 更新 `shared_encoder + fc_mu + fc_log_var` |
| **数据来源** | 从 `replay_buffer.sample_random_few_step_batch()` 采样 `(batch_size_reconstruction, time_steps)` 条目 |

#### 模块 B：`ExtendedDecoderMDP`（重建/任务分类解码器）

| 项目 | 内容 |
|---|---|
| **输入** | `(obs, action, next_obs_target, z)` — 当前状态、动作、目标下一状态、潜变量 |
| **输出** | `(state_estimate, reward_estimate, task_logits)` — 三个头的预测 |
| **训练者** | 同 `AugmentedTrainer`，与 Encoder 联合训练 |
| **损失** | `state_loss + reward_loss + task_CE_loss`，加权求和 |
| **优化器** | `optimizer_decoder` 单独更新 Decoder 参数 |

#### 模块 C：高层 `TanhGaussianPolicy`（简单环境策略）

| 项目 | 内容 |
|---|---|
| **输入** | `[obs, z]`，其中 `z` 是 Encoder 的输出 (`sample` 模式取采样值，`mean` 模式取 `[μ, log_var]`) |
| **输出** | 高层动作 `a_high ∈ [-1,1]`（在简单环境中对应子目标方向/幅度） |
| **训练者** | `PolicyTrainer.train(num_policy_steps)` |
| **损失** | SAC：Soft Q 损失 + Policy 损失 + 自动熵调节 alpha |
| **数据来源** | `replay_buffer.sample_random_few_step_batch()` 返回 encoder 数据 + sac transition 数据，**Encoder 被冻结**，只更新 Policy + Q 网络 |

#### 训练循环逻辑（每个 epoch）

```
for epoch in range(num_train_epochs):
    1. collect_replay_data()           ← 滚出数据存入 replay_buffer
    2. reconstruction_trainer.train()  ← 更新 Encoder + Decoder
    3. policy_trainer.train()          ← 更新 Policy + Q 网络（用已更新的 Encoder 提取 z）
    4. (每隔 augmented_every 轮) 做数据增强：用 Decoder 模型合成虚拟 transitions，存入 replay_buffer_augmented
    5. 评估 + snapshot 保存权重
```

---

## 三、阶段②：低层训练 (train_low_level_policy.py)

### 3.1 使用的环境

**复杂环境**（`HalfCheetahMixtureEnv`, `HopperMulti`, `WalkerMulti`, `AntMulti`），高维观测（如 cheetah obs_dim≈18），多任务（goal_front/back, forward_vel/backward_vel 等），任务向量 `task ∈ R^num_tasks`（one-hot + 数值）。

### 3.2 采样流程

```
for episode in range(epochs):
    for batch in range(batch_size):        ← 每 episode 跑 batch_size 条轨迹
        state = env.reset()
        task = env.sample_task()           ← 随机采样任务（课程学习可控制）
        while not done and j < max_traj_len:
            action = agent.choose_action(state, task)   ← PolicyNetwork(state, task) → action
            next_state, reward, done = env.step(action)
            agent.store(state, reward, done, action, next_state, task)
            agent.train(episode, save)     ← 每步后立即从 Memory 采样训练
```

### 3.3 数据保存位置

**内存中**：`SAC.memory`（`replay_memory.Memory`），存储 `Transition(state, reward, done, action, next_state, task)`，超出 `memory_size` 自动丢弃最旧数据。

**磁盘**（每 `save_after_episodes` 轮调用 `log_all()`）：

| 文件路径 | 内容 |
|---|---|
| `{path}/{experiment_name}/models/policy_model/epoch_{ep}.pth` | 低层 PolicyNetwork 完整模型 |
| `{path}/{experiment_name}/models/vf1_model/epoch_{ep}.pth` | Q 网络 1 |
| `{path}/{experiment_name}/models/vf2_model/epoch_{ep}.pth` | Q 网络 2 |
| `{path}/{experiment_name}/models/value_model/epoch_{ep}.pth` | Value 网络 |
| `{path}/{experiment_name}/config.json` | 实验配置 |
| `{path}/{experiment_name}/progress.csv` | 奖励/损失历史 |

### 3.4 各模块的输入、输出与更新方式

#### 模块：`SAC`（低层控制器 = TransferFunction）

| 项目 | 内容 |
|---|---|
| **输入** | `(state_complex, task_one_hot_value)` — 复杂环境高维观测 + 子目标任务向量 |
| **输出** | 低层连续控制动作 `a_low`（关节力矩等） |
| **训练** | 每个 env step 后 `agent.train()`：从 `Memory` 随机采 `batch_size` 条，更新 PolicyNetwork + QNetwork + ValueNetwork（标准 SAC 损失）|
| **注意** | 此模块训练时**不使用任何 Encoder/Decoder**，纯粹 task-conditioned SAC |

---

## 四、阶段③：部署推理 (task_inference_high_level_cross_agent_deployment.py)

### 4.1 模型加载

```python
encoder     = get_encoder(inference_path, ...)    ← 加载 {weights}/encoder_*.pth
simple_agent= get_simple_agent(inference_path, ...) ← 加载 {weights}/policy_*.pth
decoder     = get_decoder(inference_path, ...)    ← 加载 {weights}/decoder_*.pth
transfer_function = get_complex_agent(env, ...)   ← 加载 policy_model/epoch_X.pth
step_predictor    = SAC(...)                      ← 从头初始化（或从已有权重加载）
```

### 4.2 三种运行模式

| 模式 | `USE_TRUE_TASK` | `TRAIN_STRIDE` | 含义 |
|---|---|---|---|
| `oracle_eval` | ✅ True | ❌ False | 用真实任务标签驱动，不训练 |
| `decoder_eval` | ❌ False | ❌ False | 用 Decoder 预测任务，不训练 |
| `train_stride` | ❌ False | ✅ True | 用 Decoder 预测任务 + 在线训练 Decoder + step_predictor |

### 4.3 推理时的完整数据流（每个 high-level step）

```
complex_env.obs
    │
    ├─→ general_obs_map()
    │       └─→ simple_obs_before = [x_pos, x_vel]          (2维)
    │
contexts (n_tasks, time_steps, 2*obs_dim+1)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Encoder (DecoupledEncoder)                  │
│  输入: contexts.view(1, -1)                   │
│  输出: μ (latent_dim,), log_var (latent_dim,) │
└──────────────────┬──────────────────────────┘
                   │ μ
    ┌──────────────▼──────────────────────┐
    │  simple_agent (TanhGaussianPolicy)   │
    │  输入: [simple_obs_before, μ]        │
    │  输出: simple_action ∈ [-1, 1]       │
    └──────────────┬──────────────────────┘
                   │ simple_action
    ┌──────────────▼──────────────────────┐
    │  Decoder (ExtendedDecoderMDP)        │  (仅 USE_DECODER 模式)
    │  输入: simple_obs_before, simple_action, 0, μ  │
    │  输出: logits (num_classes,)          │
    │  → task_prediction = argmax(logits)  │
    └──────────────┬──────────────────────┘
                   │ exec_task (经过 hysteresis + lock 逻辑)
    ┌──────────────▼──────────────────────────────────────────┐
    │  Subgoal 构造                                             │
    │  alpha = 0.5*(simple_action+1) ∈ [0,1]                  │
    │  if 位置任务: x_subgoal = x_prev + α*(goal - x_prev)    │
    │  if 速度任务: v_subgoal = v_prev + α*(goal - v_prev)    │
    │  → simple_obs (one-hot + 数值) = 子目标向量              │
    └──────────────┬──────────────────────────────────────────┘
                   │ simple_obs (子目标)
    ┌──────────────▼──────────────────────┐
    │  step_predictor (SAC)               │
    │  输入: complex_obs, desired_state    │
    │  输出: sim_time_steps ∈ [1, 20]     │
    └──────────────┬──────────────────────┘
                   │ sim_time_steps
    for i in range(sim_time_steps):
    ┌──────────────▼──────────────────────┐
    │  TransferFunction (low-level policy) │
    │  输入: complex_obs, simple_obs(子目标)│
    │  输出: complex_action (关节力矩)     │
    │  complex_env.step(complex_action)    │
    └──────────────────────────────────────┘
                   │
    更新 context:
    data = [simple_obs_before, r, simple_obs_after]
    contexts = contexts[-time_steps:] (滑动窗口)
```

### 4.4 推理时的采样与数据保存

**有采样，只在 `train_stride` 模式下：**

#### 采样 ①：Decoder fine-tune 数据
```python
memory.add(task_tensor, simple_obs_tensor, simple_action_tensor, mu_tensor)
# Memory(memory_size=1e6) 中存储 Transition(task, simple_obs, simple_action, mu)
```
→ 在内存 `memory` 中，不写磁盘（除非最后保存 decoder 权重）。

#### 采样 ②：step_predictor SAC 训练数据
```python
step_predictor.store(obs_before_sim, low_level_r, done, 
                     np.array([sim_time_steps]), next_obs, desired_state)
```
→ 在 `step_predictor.memory` 中存储，纯内存。

#### 磁盘保存（推理阶段）

| 文件/目录 | 内容 |
|---|---|
| `{save_video_path}/weights/retrained_decoder.pth` | 每 5 episode 保存一次微调后的 Decoder 权重 |
| `{save_video_path}/logs/subgoals_ep{N}.csv` | 每 episode 的子目标轨迹日志 (x_before, x_after, subgoal_value, low_level_r 等) |
| `{save_video_path}/plots/decoder/finetune_loss.png` | Decoder 微调损失曲线 |
| `{save_video_path}/logs/decoder_tasks_loss/decoder_task_losses.csv` | 各任务类别的 decoder 损失 |
| `{save_video_path}/plots/step_hist/step_dist_ep{N}.png` | step_predictor 步长分布图 |
| `{save_video_path}/plots/unified_subgoal/unified_ep{N}.png` | 子目标跟踪统一可视化 |
| `{save_video_path}/Inference-trajectories/epoch_{N}.pdf` | 位置/速度轨迹图 |
| `{save_video_path}/videos/transfer_{N}.mp4` | 每 episode 渲染视频（GIF格式）|
| `{save_video_path}/tensorboard_transfer/` | Encoder 潜变量 (μ) TensorBoard Projector 数据 |
| `{save_video_path}/tensorboard_transfer_dec_input/` | Decoder 输入嵌入 (obs+μ) TensorBoard Projector 数据 |

### 4.5 推理时的模块更新（`train_stride` 模式）

#### Decoder 更新
```
每 high-level step（当 memory ≥ batch_size=32）：
  batch = memory.sample(32)   ← (task, simple_obs, simple_action, mu)
  logits = decoder(simple_obs, simple_action, 0, mu)
  loss = CrossEntropyLoss(logits, task)
  optimizer.zero_grad() → loss.backward() → optimizer.step()
  # 只有 task_decoder.last_fc, fc2, fc1 可训练（其余参数被冻结）
```

#### step_predictor 更新
```
每 high-level step：
  step_predictor.store(obs_before, low_level_r, done, [sim_steps], next_obs, desired_state)
  step_predictor.train(episode, save=False)   ← 标准 SAC 训练
```

---

## 五、数据流全景图

```
【阶段①高层训练】                        【阶段②低层训练】
简单环境                                  复杂环境
  ↓ rollout                                ↓ env.step(action)
StackedReplayBuffer                       SAC.Memory
(obs, a, r, next_obs, task_id)           (state, r, done, a, next_s, task)
  ↓ sample_random_few_step_batch           ↓ memory.sample(batch)
  ├── → Encoder.encode(context) → μ,σ     PolicyNetwork.train()
  ├── → Decoder(obs,a,z) → loss            → 保存 policy_model/epoch_X.pth
  └── → Policy(obs,z) → action (SAC)
       保存 encoder/policy/decoder_itr_X.pth
                │                                      │
                ▼                                      ▼
【阶段③部署推理】
加载 encoder + policy(simple_agent) + decoder + transfer_function(policy_model)
                │
         rolling context window
         contexts: (1, T, 2*obs+1)
                │
         Encoder → μ
         simple_agent([obs, μ]) → simple_action ∈ [-1,1]
         Decoder([obs, simple_action, μ]) → task_logits → exec_task
         subgoal = contractive interpolation (α, goal, prev_subgoal)
         step_predictor(complex_obs, subgoal) → N steps
         transfer_function(complex_obs, subgoal) × N → complex_env
                │
         更新 context，进入下一个 high-level step
                │
         (train_stride) memory → finetune Decoder + train step_predictor
```

---

## 六、关键设计要点总结

| 关键点 | 说明 |
|---|---|
| **解耦** | 高层（Encoder/Policy/Decoder）在简单环境训练，不依赖复杂动力学；低层（TransferFunction）在复杂环境训练，不依赖任务推理 |
| **接口**：高→低 | `simple_obs`（one-hot + 子目标值）是唯一接口，TransferFunction 以此为条件输出低层动作 |
| **Context 更新** | 每个 high-level step 后，将 `(simple_obs_before, r, simple_obs_after)` 追加到滑动窗口 context，下一步的 Encoder 即可获取更多历史信息 |
| **任务迟滞（hysteresis）** | 需连续 3 次 Decoder 预测同一任务才切换 `exec_task`，防止任务频繁抖动 |
| **目标锁定（goal-lock）** | 第 55 步后，若位置任务占比 ≥ 60%，则锁定该任务，防止目标任务后期漂移 |
| **收缩式子目标** | 子目标不是直接用 `simple_action`，而是 `prev + α*(goal - prev)` 的收缩插值，确保子目标单调逼近真实目标 |
| **step_predictor** | 预测需要多少个低层步才能到达子目标，本质上是一个 task-conditioned SAC，从部署时的 `(obs, desired_state, low_level_r)` 在线学习 |