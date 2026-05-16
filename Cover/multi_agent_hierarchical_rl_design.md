# 多智能体分层覆盖扫描设计

## 目标

当前工程已经验证了单智能体分层覆盖扫描：

```text
上层 PPO 输出区域 region_id
区域中心转成 guidance point
下层 PPO 根据 guidance point 执行点到点跟随
```

多智能体扩展时，推荐保持这个分层思想：

```text
上层：负责多智能体任务分配，决定每个 agent 往哪里飞
下层：负责单个 agent 的点到点跟随控制
```

下层模型可以继续使用已经训练好的统一模型。多个 agent 共享同一个低层策略，每个 agent 输入自己的观测和自己的引导点，输出自己的连续动作。

核心原则：

```text
每个 agent 应该有自己的引导点；
这些引导点应该由上层统一协调生成。
```

不建议所有 agent 共用一个引导点，因为会导致扎堆、重复覆盖和碰撞风险。

也不建议每个 agent 完全独立选点，因为容易选到相同或相邻区域，缺少协同。

---

## 总体架构

```text
Multi-Agent High-Level Allocator
        |
        | 输出每个 agent 的目标区域 / 引导点
        v
agent_0 guidance point  ---> shared low-level PPO ---> action_0 = [v, w]
agent_1 guidance point  ---> shared low-level PPO ---> action_1 = [v, w]
agent_2 guidance point  ---> shared low-level PPO ---> action_2 = [v, w]
...
        |
        v
MultiUAVCoverageEnv.step(all_actions)
```

每个高层 step 中：

1. 上层读取全局覆盖地图、所有 agent 状态、区域未覆盖信息。
2. 上层输出每个 agent 的目标区域。
3. 环境把每个区域转成对应 agent 的 guidance point。
4. 所有 agent 调用同一个低层 PPO 模型。
5. 低层执行 `option_horizon` 个微观 step。
6. 上层根据全局覆盖效果计算 reward。

---

## 阶段一：集中式高层 PPO 分配器

### 目的

第一阶段不直接上完整 MARL，而是先做一个集中式高层策略：

```text
Centralized High-Level PPO Allocator
```

它一次性输出所有 agent 的目标区域。

这个阶段的目标是验证：

```text
多 agent + 共享低层点到点模型 + 高层区域分配
```

这个机制是否有效。

### 动作空间

假设地图划分为：

```text
grid_bins x grid_bins
```

当前单智能体是：

```text
grid_bins = 5
num_regions = 25
```

多智能体可以使用：

```python
spaces.MultiDiscrete([num_regions] * num_agents)
```

例如 3 个 agent：

```python
action_space = spaces.MultiDiscrete([25, 25, 25])
```

一次高层动作：

```text
action = [4, 11, 20]
```

含义：

```text
agent_0 -> region_4
agent_1 -> region_11
agent_2 -> region_20
```

然后环境把每个 `region_id` 转成对应区域中心点：

```text
region_i -> guidance_world_i
```

### 观测空间

集中式 PPO 的 observation 可以包含全局信息：

```text
1. 当前全局覆盖率
2. 当前高层 step ratio
3. 每个 agent 的位置、朝向、速度
4. 每个区域的未覆盖比例
5. 每个区域的已扫描数量
6. 每个 agent 到每个区域中心的距离
7. 每个 agent 到每个区域中心的相对角度
```

可以先做成一个扁平向量：

```text
global_features
+ agent_features
+ region_features
+ agent_region_pair_features
```

不需要第一版就做复杂网络结构。

### 引导点设计

低层环境需要从单一引导点：

```python
self.current_guidance_points = [...]
```

改成每个 agent 一个引导点：

```python
self.agent_guidance_points = {
    0: guidance_world_0,
    1: guidance_world_1,
    2: guidance_world_2,
}
```

低层 `_get_guidance_feature(agent_idx)` 应该读取当前 agent 自己的引导点：

```python
guidance_world = self._get_agent_guidance_world(agent_idx)
```

每个 agent 调用同一个低层模型：

```python
low_actions = []
for i in range(num_agents):
    low_obs_i = low_env._get_obs()[i]
    low_action_i, _ = low_model.predict(low_obs_i, deterministic=True)
    low_actions.append(low_action_i)

low_env.step(np.array(low_actions, dtype=np.float32))
```

### Reward 设计

第一阶段建议使用全局 reward 为主：

```text
R = 覆盖率增量奖励
  + 新覆盖栅格奖励
  + 覆盖目标达成奖励
  - 重复覆盖惩罚
  - 目标冲突惩罚
  - agent 距离过近惩罚
  - 移动距离 / 能耗惩罚
  - 碰撞 / 越界惩罚
  - 每个高层 step 的时间惩罚
```

重点加入两个协同项。

#### 目标冲突惩罚

如果多个 agent 选择同一个区域，扣分：

```python
duplicate_count = num_agents - len(set(selected_regions))
target_conflict_penalty = duplicate_count * w_duplicate
```

也可以对相邻过近区域扣分：

```text
如果 region_i 和 region_j 的中心距离过近，则扣分
```

#### 重复覆盖惩罚

如果多个 agent 的探测区域高度重叠，扣分：

```text
overlap_penalty = 多 agent 当前探测圆覆盖重叠面积 / 有效地图面积
```

第一版可以近似处理，不必精确算圆面积。可以用栅格集合近似：

```python
agent_scan_sets = [...]
overlap_cells = sum_count_of_cells_seen_by_more_than_one_agent
```

### 优点

```text
实现最接近当前代码
训练稳定性较好
容易调试
可以快速验证多 agent 共享低层模型是否可行
```

### 缺点

```text
高层 action 维度随 agent 数增加
策略是集中式的，不是真正去中心化 MARL
agent 数很多时动作组合会变大
```

### 适用范围

建议先在下面规模验证：

```text
num_agents = 2 或 3
grid_bins = 5
option_horizon = 10
```

如果这个阶段效果好，再进入第二阶段。

---

## 阶段二：上层 MARL / MAPPO

### 目的

第二阶段把集中式高层 PPO 升级为真正的多智能体上层策略：

```text
每个 agent 有自己的高层 actor
每个 actor 输出自己的目标区域
训练时 critic 使用全局状态
执行时每个 agent 可以基于局部/共享信息独立决策
```

推荐算法：

```text
MAPPO
```

也就是：

```text
Centralized Training, Decentralized Execution
```

### MARL 结构

```text
actor_i(obs_i) -> region_i
centralized_critic(global_state, joint_action) -> value
```

每个 agent 的 actor 可以共享参数，也可以不共享。

对于同构无人机，建议先共享 actor 参数：

```text
所有 agent 使用同一个 high-level actor 网络
输入中包含自身 id 或相对位置信息
```

这样样本效率更高，泛化到不同 agent 数也更容易。

### 单 agent 上层观测

每个 agent 的局部观测可以包含：

```text
1. 自身位置、朝向、速度
2. 当前全局覆盖率
3. 自身附近局部覆盖地图
4. 其他 agent 的相对位置
5. 其他 agent 当前目标区域或 guidance point
6. 候选区域相对自己的距离和角度
7. 候选区域未覆盖比例
```

如果执行时允许全局通信，也可以让 actor 看到更完整的区域信息。

### Centralized critic 状态

critic 可以看全局信息：

```text
1. 全局覆盖地图
2. 所有 agent 的位置、朝向、速度
3. 所有 agent 当前 guidance point
4. 所有 agent 的上层动作
5. 全局覆盖率和 step ratio
```

critic 不负责执行，只负责训练时提供更稳定的价值估计。

### 动作空间

每个 agent 的高层动作仍然建议先保持离散区域：

```python
spaces.Discrete(num_regions)
```

所有 agent 的 joint action 是：

```text
[region_0, region_1, ..., region_n]
```

先不要直接输出连续 `(x, y)` 引导点。连续引导点更灵活，但训练难度更高，也更容易输出无效点。

### Reward 设计

MAPPO 中可以使用共享全局 reward：

```text
所有 agent 获得同一个 team reward
```

例如：

```text
R_team = 全局覆盖率增量
       + 新覆盖栅格数
       - 重复覆盖
       - 目标冲突
       - 碰撞/越界
       - 路径长度
```

也可以加入少量 individual reward：

```text
R_i = agent_i 新覆盖贡献
    - agent_i 路径代价
    - agent_i 碰撞风险
```

最终：

```text
R_i_total = R_team + alpha * R_i
```

第一版建议先用纯 team reward，逻辑更简单。

### 引导点冲突处理

即使用 MARL，也建议保留一个环境级冲突约束：

```text
如果多个 agent 选中同一区域，允许但扣分
```

不要直接强行改动作，否则会让策略难以理解环境反馈。

可以在训练后期再加入规则修正：

```text
如果两个 agent 选同一区域，给距离更近的 agent 保留，另一个改分配到相邻未覆盖区域
```

但第一版不建议这样做，先让 reward 驱动策略自己学会分散。

### 优点

```text
更符合多智能体协同问题
可以支持 agent 数量增加
执行时可以去中心化
更容易学习动态协同和避让分工
```

### 缺点

```text
工程复杂度明显更高
训练稳定性弱于集中式 PPO
需要 MARL 框架或自定义训练循环
reward shaping 更敏感
debug 难度更高
```

---

## 推荐落地路线

### Step 1：扩展低层环境为多 agent guidance

修改低层环境，使每个 agent 都能读取自己的引导点：

```text
agent_guidance_points[agent_id] -> guidance_world
```

确保已有共享低层 PPO 可以被多个 agent 调用。

### Step 2：实现集中式高层 wrapper

新增类似：

```text
MultiAgentHighLevelGuidanceEnv
```

内部逻辑：

```text
MultiDiscrete action
-> 每个 action 转成 region center
-> 写入每个 agent 的 guidance
-> 下层共享模型执行 option_horizon 步
-> 返回高层全局 obs 和 team reward
```

### Step 3：训练集中式高层 PPO

先用：

```text
num_agents = 2
grid_bins = 5
option_horizon = 10
```

验证指标：

```text
覆盖率是否提升
达到 98% 所需低层 step 是否减少
重复覆盖是否下降
碰撞是否可控
agent 是否出现扎堆
```

### Step 4：扩大到 3 个 agent

如果 2 个 agent 稳定，再扩到 3 个 agent。

观察：

```text
是否出现目标冲突
是否大量重复覆盖
是否因为低层共享模型导致互相干扰
```

### Step 5：升级 MAPPO

集中式 PPO 验证有效后，再做 MAPPO：

```text
shared high-level actor
centralized critic
team reward
shared low-level controller
```

这个阶段重点不是重新训练低层，而是让上层学会协同分配。

---

## 最小代码改造方向

当前单智能体逻辑：

```python
target_world = self._action_to_world(action)
self.low_env.current_guidance_points = [
    {
        "grid": (...),
        "world": (float(target_world[0]), float(target_world[1])),
        "score": 0.0,
    }
]
```

多智能体第一阶段可以改成：

```python
for agent_id, region_id in enumerate(action):
    target_world = self._action_to_world(int(region_id))
    self.low_env.agent_guidance_points[agent_id] = {
        "grid": self._world_to_grid(target_world),
        "world": (float(target_world[0]), float(target_world[1])),
        "score": 0.0,
    }
```

低层执行：

```python
low_actions = []
for agent_id in range(self.low_env.n):
    low_obs = self.low_env._get_obs()[agent_id]
    low_action, _ = self.low_level_model.predict(low_obs, deterministic=True)
    low_actions.append(low_action)

self.low_env.step(np.array(low_actions, dtype=np.float32))
```

低层 guidance feature：

```python
def _get_guidance_feature(self, agent_idx: int):
    guidance_world = self._get_agent_guidance_world(agent_idx)
    ...
```

---

## 结论

你的设想是合理的：

```text
上层使用 MARL / 分配器输出每个 agent 的引导点；
下层使用已经训练好的共享点到点 PPO 模型；
每个 agent 调用同一个低层模型执行自己的 guidance point。
```

推荐不要一步到位上复杂 MARL，而是分两阶段：

```text
阶段一：集中式高层 PPO + MultiDiscrete 动作，验证多 agent 分配机制
阶段二：MAPPO 上层 MARL，训练更通用的多 agent 协同策略
```

这样既符合当前工程结构，也能降低多智能体训练的风险。
