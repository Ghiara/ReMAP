from third_party.SAC.sac_envs.walker import WalkerGoal
from third_party.SAC.sac_envs.hopper import HopperGoal
import numpy as np

pi = 3.141592

config = dict(
    # 总训练轮数（epoch 数）
    epochs = 3000,

    # 每条轨迹初始最大步数（会被 curriculum.max_steps 动态覆盖）
    max_traj_len = 500,

    # 经验回放池容量（越大越不容易遗忘旧分布，但占内存更多）
    memory_size = 1e+6,

    # 每个 epoch 采样多少条轨迹（环境 rollout 数）
    batch_size = 20,

    # 每次从 replay buffer 取多少条样本做一次梯度更新
    batch_size_memory = 256,

    # 每个 epoch 内策略/价值网络更新步数
    policy_update_steps = 2048,

    # 折扣因子，越接近 1 越重视长期回报
    gamma = 0.99,

    # SAC 温度系数，控制探索熵权重
    alpha = 0.2,

    # 学习率（actor/critic 共用）
    lr = 3e-4,

    # 奖励缩放系数（传给 SAC）
    reward_scale = 1,

    # 目标跟踪任务幅度范围 [min, max]，单位米（goal_front/goal_back 使用）
    max_goal = [2.0, 15.0],

    # 跳跃任务范围（当前四任务训练不主要使用）
    max_jump = [1.5, 3.],

    # 旋转姿态任务范围（当前四任务训练不主要使用）
    max_rot = [pi / 6., pi / 2.],

    # 速度跟踪任务幅度范围 [min, max]，单位 m/s（forward_vel/backward_vel 使用）
    max_vel = [0.5, 4.0],

    # 旋转角速度任务范围（当前四任务训练不主要使用）
    max_rot_vel = [2. * pi, 4. * pi],

    # 环境名（训练脚本根据它实例化 HopperMulti）
    env = 'hopper_multi',

    # 本次实验名（输出目录名）
    experiment_name = 'hopper_multi_new_config_v3_cleanup_test',

    # task 向量维度上限（通常等于 max(tasks.values()) + 1）
    task_dim = 5,

    # Actor 隐藏层结构
    hidden_layers_actor = [300,300,300,300],

    # Critic 隐藏层结构
    hidden_layers_critic = [300,300,300,300],

    # 每隔多少 epoch 保存一次模型与日志
    save_after_episodes = 10,

    # 绘图/可视化间隔（被日志模块使用）
    plot_every = 5,

    reward_params = dict(
        velocity_ctrl_weight = 1.2e-3,
        goal_ang_limit = 0.4,
        forward_ang_limit_base = 0.44,
        forward_ang_limit_bonus = 0.12,
        backward_ang_limit_base = 0.50,
        backward_ang_limit_bonus = 0.10,
    ),

    task_sampling_weights = dict(
        forward_vel = 2.4,
        backward_vel = 2.4,
        goal_front = 1.0,
        goal_back = 1.0,
    ),

    # 任务到 task 向量索引的映射
    # goal_front / goal_back: 位置目标
    # forward_vel / backward_vel: 速度目标
    tasks = dict(
                forward_vel=2, backward_vel=3,
                goal_front=0, goal_back=1,
                ),

    # 启用 random_reset 后，初始 x 位置采样范围
    random_reset_pos_range = [-8.0, 8.0],

    # 启用 random_reset 后，初始 x 速度采样范围
    random_reset_vel_range = [-0.5, 0.5],

    curriculum = dict(
        # 速度任务解锁完整难度的 epoch 阈值
        # 训练代码中：episode < max_vel 时 velocity task 会降难（目标速度减半）
        max_vel=250,

        # 轨迹内任务切换课程的阶段边界（epoch）
        # 超过某个值后，会采用对应 changes_per_trajectory
        change_tasks_after = [200, 350, 550],

        # 每条轨迹内大约切换任务的次数
        # 0 表示整条轨迹只做一个任务；值越大，多任务切换越频繁、训练越难
        changes_per_trajectory = [0, 1, 3],

        # 轨迹最大步数课程的阶段边界（epoch）
        max_steps_epochs = [150,350,700],

        # 对应阶段使用的 max_traj_len
        # 当前设置含义：先 500，之后提升到 1000，并保持长轨迹训练
        max_steps = [500,1000,1000],

        # 从该 epoch 开始启用 random_reset（随机初始位置/速度）
        random_initialization = 35000,
    ),


)
