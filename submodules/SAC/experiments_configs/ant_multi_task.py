import numpy as np

pi = 3.141592

config = dict(
    # 总训练轮数（epoch 数）
    epochs=3000,

    # 每条轨迹初始最大步数（会被 curriculum.max_steps 动态覆盖）
    # 先用更短轨迹让 ant 学会基础稳定控制，再逐步加长时域
    max_traj_len=200,

    # 经验回放池容量（越大越不容易遗忘旧分布，但更占内存）
    memory_size=1e6,

    # 每次从 replay buffer 采样多少条数据做一次更新
    batch_size_memory=256,

    # 每个 epoch 采样多少条轨迹（环境 rollout 数）
    batch_size=20,

    # 每个 epoch 内 actor/critic 的更新步数
    policy_update_steps=2048,

    # 折扣因子，越接近 1 越重视长期回报
    gamma=0.99,

    # SAC 温度系数，控制熵正则强度（探索程度）
    alpha=0.2,

    # 学习率（actor/critic 共用）
    lr=3e-4,

    # 奖励缩放系数（传给 SAC）
    reward_scale=1,

    # 位置目标范围 [min, max]（单位米），用于 goal_front / goal_back
    max_goal=[1.0, 10.0],

    # 跳跃任务范围（当前四任务训练不主要使用）
    max_jump=[1.5, 3.0],

    # 姿态角范围（当前四任务训练不主要使用）
    max_rot=[pi / 6., pi / 2.],

    # 速度目标范围 [min, max]（单位 m/s），用于 forward_vel / backward_vel
    max_vel=[0.4, 2.5],

    # 角速度范围（当前四任务训练不主要使用）
    max_rot_vel=[2. * pi, 4. * pi],

    # 环境名（训练脚本据此实例化 AntMulti）
    env='ant_multi',

    # 实验名（输出目录名）
    experiment_name='ant_multi_new_config_v4_run2',

    # task 向量维度上限（通常等于 max(tasks.values()) + 1）
    task_dim=4,

    # Actor 网络隐藏层结构
    hidden_layers_actor=[300, 300, 300, 300],

    # Critic 网络隐藏层结构
    hidden_layers_critic=[300, 300, 300, 300],

    # 每隔多少个 epoch 保存一次模型与训练曲线
    save_after_episodes=10,

    # 日志绘图间隔（由可视化模块使用）
    plot_every=10,

    # 任务名到 task 向量索引的映射
    # goal_front / goal_back: 位置目标跟踪
    # forward_vel / backward_vel: 速度目标跟踪
    tasks=dict(
        goal_front=0,
        goal_back=1,
        forward_vel=2,
        backward_vel=3,
    ),

    # Ant 稳定低层控制：先让 reset 从正常站立高度开始，减少初始接触爆冲
    initial_torso_height=0.72,
    reset_xy_noise=0.03,
    reset_joint_noise=0.03,
    reset_height_noise=0.01,
    reset_velocity_noise=0.03,

    # 四个任务均衡采样，略微增加 backward 任务，避免 goal_back 训练不足
    task_sample_weights=dict(
        goal_front=1.0,
        goal_back=1.25,
        forward_vel=1.0,
        backward_vel=1.25,
    ),

    # 启用 random_reset 后，初始 x 位置采样范围
    random_reset_pos_range=[-8.0, 8.0],

    # 启用 random_reset 后，初始 x 速度采样范围
    random_reset_vel_range=[-0.5, 0.5],

    curriculum=dict(
        # 速度任务解锁完整难度的 epoch 阈值
        # 训练代码中：episode < max_vel 时 velocity task 会降难（目标速度减半）
        max_vel=150,

        # 轨迹内任务切换课程的阶段边界（epoch）
        # 超过某个阈值后，会采用对应 changes_per_trajectory
        change_tasks_after=[200, 450, 800],

        # 每条轨迹内切换任务次数
        # 0 表示整条轨迹一个任务；值越大，训练更难但切换适应更强
        changes_per_trajectory=[0, 1, 2],

        # 轨迹长度课程阶段边界（epoch）
        # 训练循环里是 episode > threshold 才切换，所以：
        # 1-150: 200 步
        # 151-350: 300 步
        # 351-700: 500 步
        # 701+: 700 步
        max_steps_epochs=[150, 400, 800],

        # 各阶段对应 max_traj_len（只增不减）
        # 先让 ant 在短时域内学会走稳，再慢慢增加长期 credit assignment 难度
        max_steps=[250, 350, 500],

        # 从该 epoch 开始启用 random_reset
        # 你当前设得很大，等效于训练期间基本关闭 random_reset
        random_initialization=1200,
    ),

    # 旧逻辑保留项（当前训练主循环基本未使用）
    random_restart_after=1,

    # 预训练模型入口（None 表示从头训练）
    use_termination_after=80,
    termination_ramp_epochs=320,

    # 更强地鼓励四足贴地、躯干稳定，不奖励跳高刷进度
    healthy_reward=0.55,
    termination_penalty=-8.0,
    target_torso_height=0.70,
    min_torso_height=0.34,
    max_torso_height=1.05,
    min_upright=0.55,
    max_lateral_offset=1.20,
    max_vertical_velocity=1.80,
    max_pitch_rate=7.0,
    max_state_abs=100.0,
    height_penalty_weight=1.0,
    upright_penalty_weight=1.0,
    vertical_velocity_penalty_weight=0.16,
    pitch_rate_penalty_weight=0.035,
    action_smooth_penalty_weight=0.006,
    goal_progress_weight=0.8,
    goal_ctrl_cost=8e-3,
    vel_ctrl_cost=8e-3,

    pretrained=None,
)
