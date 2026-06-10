from third_party.SAC.sac_envs.walker import WalkerGoal
from third_party.SAC.sac_envs.hopper import HopperGoal
import numpy as np

pi = 3.141592

config = dict(
    # 总训练轮数（epoch 数）
    epochs = 3000,

    # 每条轨迹初始最大步数（会被 curriculum.max_steps 动态覆盖）
    max_traj_len = 300,

    # 经验回放池容量（越大越不容易遗忘旧分布，但占内存更多）
    memory_size = 1e+6,

    # 每个 epoch 采样多少条轨迹（环境 rollout 数）
    batch_size = 20,

    # 每次从 replay buffer 采样多少条数据做一次更新
    batch_size_memory = 128,

    # 每个 epoch 内 actor/critic 的更新步数
    policy_update_steps = 1024,

    # 折扣因子，越接近 1 越看重长期回报
    gamma = 0.99,

    # SAC 温度系数，控制策略熵项权重
    alpha = 0.2,

    # 学习率（actor/critic 共用）
    lr = 3e-4,

    # 奖励缩放系数（传给 SAC）
    reward_scale = 1,

    # 位置目标范围 [min, max]（单位米），用于 goal_front / goal_back
    max_goal = [2.0, 15.0],

    # 速度目标范围 [min, max]（单位 m/s），用于 forward_vel / backward_vel
    max_vel = [0.5, 4.0],

    # 站立/姿态类任务的角度范围（当前四任务训练不主要使用）
    max_rot = [pi / 6., pi / 2.],

    # 旋转角速度范围（当前四任务训练不主要使用）
    max_rot_vel = [2. * pi, 4. * pi],

    # 环境名（训练脚本据此实例化 WalkerMulti）
    env = 'walker_multi',

    # 实验名（输出目录和日志前缀）
    experiment_name = 'walker_multi_new_config_v0_run2',

    # task 向量维度上限（一般等于 max(tasks.values()) + 1）
    task_dim = 5,

    # 任务名到 task 向量索引的映射
    # goal_front / goal_back: 位置跟踪任务
    # forward_vel / backward_vel: 速度跟踪任务
    tasks = dict(
                forward_vel=2, backward_vel=3,
                goal_front=0, goal_back=1,
                 ),

    # Actor 网络隐藏层结构
    hidden_layers_actor = [300,300,300,300],

    # Critic 网络隐藏层结构
    hidden_layers_critic = [300,300,300,300],

    # 每隔多少个 epoch 保存一次模型与训练曲线
    save_after_episodes = 20,

    # 日志绘图间隔（由可视化模块使用）
    plot_every = 100,

    # random_reset 启用后，初始 x 位置采样范围
    random_reset_pos_range = [-10.0, 10.0],

    # random_reset 启用后，初始 x 速度采样范围
    random_reset_vel_range = [-0.5, 0.5],

    curriculum = dict(
        # 速度任务解锁完整难度的 epoch 阈值
        # 训练代码中：episode < max_vel 时 velocity task 会降难（目标速度减半）
        max_vel=150,

        # 轨迹内任务切换课程的阶段边界（epoch）
        # 超过某个阈值后，会使用对应的 changes_per_trajectory
        change_tasks_after = [150,250,400],

        # 每条轨迹内切换任务的次数
        # 0 表示整条轨迹一个任务；值越大，任务切换越频繁、训练越难
        changes_per_trajectory = [0,2,4],

        # 轨迹最大步数课程的阶段边界（epoch）
        max_steps_epochs = [150,350,700],

        # 每个阶段对应的 max_traj_len
        # 注意：这里会影响你看到的 traj_len 曲线是否变长/变短
        max_steps = [300,500,700],

        # 从该 epoch 开始启用 random_reset（随机初始状态）
        random_initialization = 30000,
    ),

)
