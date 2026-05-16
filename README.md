# UAV Coverage RL — Stable Baselines3

基于 Stable Baselines3 的无人机覆盖扫描强化学习工程，采用分层架构：上层 PPO 负责区域分配，下层 PPO 负责点到点跟随控制。

---

## 目录结构

```
stable_baseline3/
├── Cover/                  # 单智能体分层 RL（已验证）
│   ├── env.py              # 单/多 UAV 底层仿真环境
│   ├── high_level_env.py   # 单智能体上层 Gym wrapper
│   ├── train_high_level.py # 上层训练脚本
│   └── model_test_high_level.py  # 推理 / 可视化
│
├── Cover_multi/            # 多智能体集中式分层 RL（阶段一）
���   ├── env.py              # 多 UAV 底层仿真环境
│   ├── multi_high_level_env.py   # 集中式多智能体上层 wrapper
│   └── train_multi_high_level.py # 多智能体上层训练脚本
│
├── Cover_logic/            # 基于规则引导点的对比基线
│   ├── env.py
│   ├── global_guidance.py  # 规则生成引导点
│   └── train.py
│
└── PID/                    # PID 控制基线
    ├── env.py
    └── train.py
```

---

## 架构概览

### 分层结构

```
上层 PPO（区域分配）
    输出每个 agent 的目标区域 region_id
        ↓
    region_id → 区域中心世界坐标 guidance_point
        ↓
下层 PPO（点到点跟随，共享模型）
    每个 agent 独立调用，输入自身观测 + guidance_point
    输出连续动作 [v, w]（线速度 + 角速度）
        ↓
MultiUAVCoverageEnv.step(all_actions)
```

### 上层决策节奏

每个上层 `step()` 包含两个退出条件：

- 下层执行步数达到 `option_horizon`（默认 15 步）
- **任意 agent 到达自己引导点的距离 < `early_reach_dist`（默认 4m）**，提前结束当前 option，立即触发新的上层决策

---

## 模块说明

### Cover_multi/multi_high_level_env.py

集中式多智能体上层环境，核心类 `MultiAgentHighLevelEnv`。

**动作空间**

```python
spaces.MultiDiscrete([num_regions] * num_agents)
# 例：3 个 agent，25 个区域 → MultiDiscrete([25, 25, 25])
```

**观测空间**（扁平向量）

| 分组 | 维度 | 内容 |
|------|------|------|
| 全局 | 2 | 覆盖率、step ratio |
| 每个 agent | 7 × n | 位置、朝向、速度、最近邻距离、上轮区域 |
| 每个区域 | 2 × R | 未覆盖比例、已扫描栅格数 |
| agent-region 对 | 2 × n × R | 归一化距离、相对角度 |

**奖励组成**

| 项目 | 方向 | 权重 | 说明 |
|------|------|------|------|
| `w_cov * delta_cov` | + | 8.0 | 全局覆盖率增量 |
| `w_new * delta_new` | + | 1.0 | 新覆盖栅格增量（归一化） |
| `reach_bonus` | + | 0.15/agent | 从远处到达引导点 |
| `r_cov_goal` | + | env_cfg | 达到覆盖目标一次性奖励 |
| `w_travel` | - | 0.1 | 平均移动距离惩罚 |
| `w_fail` | - | 2.0 | 碰撞障碍物或越界 |
| `w_agent_collision` | - | 2.0 | agent 间碰撞 |
| `w_duplicate` | - | 0.3/次 | 多 agent 选同一区域 |
| `w_target_near` | - | 0.1 | 目标点间距 < `min_target_dist`(6m) |
| `w_switch` | - | 0.03/次 | 切换目标区域 |
| `w_scan_overlap` | - | 0.4 | 探测圆实时重叠比例 |
| `w_agent_proximity` | - | 0.2 | agent 间距 < `min_agent_dist`(2m) |
| `w_fully_scanned` | - | 0.5/agent | 目标区域已被完全扫描 |
| `time_penalty` | - | 0.03 | 每步固定时间惩罚 |

**关键配置参数（MultiHighLevelConfig）**

```python
max_high_steps: int = 80       # 上层最大宏观步数
option_horizon: int = 15       # 每次决策下层最多执行步数
grid_bins: int = 5             # 地图划分粒度（5×5=25 区域）
reach_dist: float = 4.0        # reach_bonus 判定距离（m）
early_reach_dist: float = 4.0  # 提前结束 option 的距离阈值（m）
min_target_dist: float = 6.0   # 目标点过近惩罚阈值（m）
min_agent_dist: float = 2.0    # agent 过近惩罚阈值（m）
```

---

## 快速开始

### 1. 训练下层单智能体模型

```bash
cd Cover
python train.py
```

下层模型保存至 `Cover/low_model/ppo_model_save.zip`（路径可在脚本中配置）。

### 2. 训练多智能体上层模型

```bash
cd Cover_multi
python train_multi_high_level.py \
    --low_model low_model/ppo_model_save.zip \
    --num_agents 3 \
    --n_envs 8 \
    --total_timesteps 2000000
```

模型和 TensorBoard 日志保存至 `Cover_multi/check_point_multi_high_level/version_N/`。

**常用参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--low_model` | `low_model/ppo_model_save.zip` | 下层模型路径 |
| `--num_agents` | 3 | 智能体数量 |
| `--n_envs` | 8 | 并行环境数 |
| `--total_timesteps` | 2,000,000 | 总训练步数 |
| `--grid_bins` | 5 | 区域划分粒度 |
| `--option_horizon` | 15 | 每次决策下层步数 |
| `--max_high_steps` | 80 | 上层最大宏观步数 |
| `--use_subproc` | False | 使用多进程并行（SubprocVecEnv） |

### 3. 查看训练曲线

```bash
tensorboard --logdir Cover_multi/check_point_multi_high_level/version_N/tensorboard
```

---

## 设计说明

### 为什么分层

下层点到点跟随已经验证有效，上层只需学习"去哪里"而不是"怎么飞"，大幅降低上层策略的探索难度。

### 为什么用集中式 PPO（阶段一）

- 实现最接近单智能体结构，工程风险低
- 训练稳定，便于调试
- 先验证多 agent 共享下层模型的可行性

阶段二计划升级为 MAPPO（集中训练、分散执行），参见 `Cover_multi/multi_agent_hierarchical_rl_design.md`。

### 已扫描区域惩罚

上层 action 指向已被完全扫描的区域时触发 `w_fully_scanned` 惩罚，引导策略主动探索未覆盖区域，避免无效重复扫描。

### 提前退出机制

任意 agent 到达引导点（距离 < `early_reach_dist`）时立即结束当前 option，上层重新决策，提高宏观决策频率，减少 agent 在已到达目标附近原地徘徊的时间浪费。
