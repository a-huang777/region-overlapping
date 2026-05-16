import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class Config:
    """环境与训练参数配置类 (纯二维版本)"""
    # 环境基础参数
    env_size = 50.0  # 环境边长 50m x 50m
    grid_res = 2.0  # 栅格分辨率 1.0m（单栅格更大，计算量更低）
    grid_size = int(env_size / grid_res)  # 50 x 50 栅格
    max_steps = 300  # 最大步数
    infer_mode = False  # 推理模式
    # 无人机参数
    num_agents = 1
    uav_radius = 0.5  # 无人机碰撞半径
    det_radius = 8.0  # 探测半径（增大扫描范围，降低任务难度）
    safe_radius = 3.0  # 起点安全区半径
    start_spawn_ratio = 0.9  # 起点采样区域边长占比（50x50 时 0.9 -> 中心 45x45）

    # 运动学边界
    v_max = 2.0  # 最大线速度 m/s
    w_max = np.pi / 4  # 最大角速度 rad/s

    # 障碍物参数
    num_obstacles = 0  # 障碍物数量
    obs_radius_range = [4.0, 4.0]  # 障碍物半径范围
    obs_min_gap = 5.0  # 障碍物边缘之间的最小通行距离

    # 感知参数
    num_lidar_rays = 16  # 激光雷达射线数
    lidar_range = 10.0  # 雷达最大量程
    local_map_size = 10  # 局部覆盖地图维度
    critic_map_channels = 3  # centralized critic 的全局地图通道数

    # 奖励函数系数
    c_step = 0.01  # 步数惩罚
    c_smooth = 0.02  # 动作平滑惩罚
    c_cov = 0.04  # 有效覆盖奖励 (每覆盖一个新栅格)
    c_overlap = 0.0002  # 覆盖重叠惩罚
    c_covrate = 1.0  # 覆盖率奖励系数 (鼓励持续提升全局覆盖率)
    c_guidance = 0.2  # 引导点距离差奖励系数（动作后更接近则奖励为正）

    # 终止态绝对奖励
    r_col = -2.0  # 碰撞极大惩罚
    r_cov_goal = 5.0  # 覆盖目标达成奖励
    coverage_goal = 0.98  # 覆盖率达到该阈值即任务完成（降低达标难度）


class MultiUAVCoverageEnv(gym.Env):
    """多无人机协同覆盖扫描环境 (2D)"""

    def __init__(self, cfg=Config()):
        super(MultiUAVCoverageEnv, self).__init__()
        self.cfg = cfg
        self.n = cfg.num_agents

        # 动作空间: [v, w] 连续动作
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n, 2), dtype=np.float32)

        # 状态空间维度计算:
        # 自身(5) + 雷达(16) + 队友((n-1)*4) + 局部地图(100) + 引导点特征(2: 距离/角度)
        obs_dim = 5 + cfg.num_lidar_rays + (self.n - 1) * 4 + cfg.local_map_size ** 2 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n, obs_dim), dtype=np.float32)

        # 环境内部状态变量初始化 (纯二维)
        self.agents_pos = np.zeros((self.n, 2))  # x, y
        self.agents_theta = np.zeros(self.n)
        self.agents_vel = np.zeros((self.n, 2))  # v, w
        self.target_pos = np.zeros(2)
        self.start_pos = np.zeros(2)

        # 障碍物列表与全局覆盖地图
        self.obstacles = []
        self.global_map = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.int8)
        self.steps = 0
        self.episode_idx = 0
        # 引导点由外部上层模块提供；env 内部不再自动生成引导点。
        self.current_guidance_points = []
        self.current_guidance_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_idx += 1
        self.steps = 0
        self.global_map.fill(0)

        # 1. 生成起点：在地图中心区域内随机采样
        # 例如 env_size=50 且 start_spawn_ratio=0.9 时，采样范围为中心 45x45 区域
        ratio = float(np.clip(self.cfg.start_spawn_ratio, 0.1, 1.0))
        center = self.cfg.env_size / 2.0
        half_span = 0.5 * self.cfg.env_size * ratio
        low = center - half_span
        high = center + half_span
        self.start_pos = np.array(
            [
                np.random.uniform(low, high),
                np.random.uniform(low, high),
            ],
            dtype=np.float32,
        )
        # 保留 target_pos 字段以兼容历史可视化脚本，不参与任务定义
        self.target_pos = self.start_pos.copy()

        # 2. 初始化无人机位置 (二维平面散开)
        for i in range(self.n):
            angle = i * (2 * np.pi / self.n)
            r = 1.0
            self.agents_pos[i, 0] = self.start_pos[0] + r * np.cos(angle)
            self.agents_pos[i, 1] = self.start_pos[1] + r * np.sin(angle)
            self.agents_theta[i] = angle
        self.agents_vel.fill(0.0)

        # 3. 静态圆形障碍物生成逻辑 (均匀分布 + 最小间距 + 动态半径)
        self.obstacles = []
        max_retries = 100

        for _ in range(self.cfg.num_obstacles):
            for _ in range(max_retries):
                orad = np.random.uniform(self.cfg.obs_radius_range[0], self.cfg.obs_radius_range[1])
                ox = np.random.uniform(orad, self.cfg.env_size - orad)
                oy = np.random.uniform(orad, self.cfg.env_size - orad)

                valid_position = True

                dist_to_start = np.hypot(ox - self.start_pos[0], oy - self.start_pos[1])
                min_req_dist = self.cfg.safe_radius + orad + self.cfg.obs_min_gap
                if dist_to_start < min_req_dist:
                    valid_position = False
                    continue

                for ex, ey, erad in self.obstacles:
                    dist_to_obs = np.hypot(ox - ex, oy - ey)
                    if dist_to_obs < (orad + erad + self.cfg.obs_min_gap):
                        valid_position = False
                        break

                if valid_position:
                    self.obstacles.append((ox, oy, orad))
                    grid_x = int(ox / self.cfg.grid_res)
                    grid_y = int(oy / self.cfg.grid_res)
                    grid_rad = int(orad / self.cfg.grid_res)
                    for gx in range(max(0, grid_x - grid_rad), min(self.cfg.grid_size, grid_x + grid_rad)):
                        for gy in range(max(0, grid_y - grid_rad), min(self.cfg.grid_size, grid_y + grid_rad)):
                            if (gx - grid_x) ** 2 + (gy - grid_y) ** 2 <= grid_rad ** 2:
                                self.global_map[gx, gy] = -1
                    break

        self._update_coverage()
        self.current_guidance_points = []
        self.current_guidance_idx = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        rewards = np.zeros(self.n)
        dones = np.zeros(self.n, dtype=bool)
        info = {f'agent_{i}': {} for i in range(self.n)}
        pre_guidance_world = self._get_active_guidance_world()

        prev_pos = self.agents_pos.copy()
        prev_vel = self.agents_vel.copy()
        prev_covered_grids = np.sum(self.global_map == 1)

        # 1. 运动学更新 (二维)
        for i in range(self.n):
            a_v, a_w = action[i]

            v = ((a_v + 1.0) / 2.0) * self.cfg.v_max
            w = a_w * self.cfg.w_max
            # print(f"[step={self.steps}] agent_{i}: v={v:.3f}, w={w:.3f}")

            self.agents_vel[i, 0] = v
            self.agents_vel[i, 1] = w
            self.agents_theta[i] += w
            self.agents_pos[i, 0] += v * np.cos(self.agents_theta[i])
            self.agents_pos[i, 1] += v * np.sin(self.agents_theta[i])

        # 2. 更新覆盖率并获取 N_new, N_old
        cov_stats = self._update_coverage()

        total_valid_grids = np.sum(self.global_map >= 0)
        covered_grids = np.sum(self.global_map == 1)
        coverage_rate = covered_grids / max(total_valid_grids, 1)
        delta_covered_grids = covered_grids - prev_covered_grids
        coverage_goal_reached = coverage_rate >= self.cfg.coverage_goal
        collision_any = False

        # 3. 计算奖励与终止条件
        for i in range(self.n):
            r_components = {}

            # (1) 步数惩罚
            r_step = -self.cfg.c_step
            r_components['step'] = r_step

            # (2) 动作平滑惩罚
            r_smooth = -self.cfg.c_smooth * (
                        (self.agents_vel[i, 0] - prev_vel[i, 0]) ** 2 + (self.agents_vel[i, 1] - prev_vel[i, 1]) ** 2)
            r_components['smooth'] = r_smooth

            # (3) 协同覆盖奖励
            n_new, n_old = cov_stats[i]
            r_cov = self.cfg.c_cov * n_new
            r_overlap = -self.cfg.c_overlap * n_old
            r_components['cov'] = r_cov
            r_components['overlap'] = r_overlap
            # 使用“全局覆盖增量”替代“全局覆盖率绝对值”，奖励更符合扫描任务目标
            r_components['cov_delta'] = self.cfg.c_covrate * delta_covered_grids*0.1
            if pre_guidance_world is not None:
                prev_dist = self._distance_to_guidance(prev_pos[i], pre_guidance_world)
                post_dist = self._distance_to_guidance(self.agents_pos[i], pre_guidance_world)
                r_guidance = self.cfg.c_guidance * (prev_dist - post_dist)
            else:
                prev_dist = 0.0
                post_dist = 0.0
                r_guidance = 0.0
            r_components['guidance_dist'] = r_guidance

            # (4) 碰撞与覆盖达标检测
            r_terminal = 0.0
            col_flag = False

            x, y = self.agents_pos[i, 0], self.agents_pos[i, 1]
            if x < 0 or x > self.cfg.env_size or y < 0 or y > self.cfg.env_size:
                col_flag = True

            for ox, oy, orad in self.obstacles:
                if np.hypot(x - ox, y - oy) < (self.cfg.uav_radius + orad):
                    col_flag = True
                    break

            if col_flag:
                r_terminal = self.cfg.r_col
                dones[i] = True
                collision_any = True
                if self.cfg.infer_mode:
                    print(f"collision at step {self.steps}")
            elif coverage_goal_reached:
                # 覆盖达标后提前结束，提高训练效率
                r_terminal = self.cfg.r_cov_goal
                dones[i] = True
                if self.cfg.infer_mode:
                    print(f"coverage goal reached at step {self.steps}")
            r_components['terminal'] = r_terminal

            # 单步总奖励加总与截断
            total_r = sum(r_components.values())
            # rewards[i] = np.clip(total_r, -2.0, 2.0)
            rewards[i] = total_r

            info[f'agent_{i}'] = r_components
            info[f'agent_{i}']['clipped_total'] = rewards[i]
            info[f'agent_{i}']['coverage_rate'] = coverage_rate
            info[f'agent_{i}']['delta_covered_grids'] = delta_covered_grids
            info[f'agent_{i}']['covered_grids'] = covered_grids
            info[f'agent_{i}']['total_valid_grids'] = total_valid_grids
            info[f'agent_{i}']['guidance_dist_prev'] = prev_dist
            info[f'agent_{i}']['guidance_dist_post'] = post_dist

        truncated = self.steps >= self.cfg.max_steps
        done_all = collision_any or coverage_goal_reached
        # 4. 引导点由外部上层更新：这里不做自动刷新/切换。
        active_guidance = self._get_active_guidance_point()
        guidance_points = [active_guidance] if active_guidance is not None else []
        # if guidance_points:
        #     print(
        #         f"[Episode {self.episode_idx} Step {self.steps}] guidance_points: "
        #         f"{[p['grid'] for p in guidance_points]}"
        #     )
        # else:
        #     print(f"[Episode {self.episode_idx} Step {self.steps}] guidance_points: []")

        for i in range(self.n):
            info[f'agent_{i}']['guidance_points'] = guidance_points

        if truncated and self.cfg.infer_mode:
            print(f"steps truncated")
            print(f"[Episode {self.episode_idx}] end coverage rate: "
                  f"{coverage_rate * 100.0:.2f}% ({covered_grids}/{total_valid_grids})")

        if done_all and self.cfg.infer_mode:
            print(f"[Episode {self.episode_idx}] end coverage rate: "
                  f"{coverage_rate * 100.0:.2f}% ({covered_grids}/{total_valid_grids})")
        return self._get_obs(), rewards, done_all, truncated, info

    def _update_coverage(self):
        """更新全局栅格地图并返回每架无人机的新/旧覆盖网格数"""
        stats = []
        for i in range(self.n):
            n_new, n_old = 0, 0
            x, y = self.agents_pos[i, 0], self.agents_pos[i, 1]
            gx = int(x / self.cfg.grid_res)
            gy = int(y / self.cfg.grid_res)
            grad = int(self.cfg.det_radius / self.cfg.grid_res)

            for dx in range(-grad, grad + 1):
                for dy in range(-grad, grad + 1):
                    if dx ** 2 + dy ** 2 <= grad ** 2:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.cfg.grid_size and 0 <= ny < self.cfg.grid_size:
                            cell_val = self.global_map[nx, ny]
                            if cell_val == 0:
                                n_new += 1
                                self.global_map[nx, ny] = 1  # 标记为已覆盖
                            elif cell_val == 1:
                                n_old += 1
            stats.append((n_new, n_old))
        return stats

    def _get_obs(self):
        """构造归一化的局部观测状态 (纯二维)"""
        obs = []
        for i in range(self.n):
            # 1. 自身状态归一化
            x, y = self.agents_pos[i, 0], self.agents_pos[i, 1]
            v, w = self.agents_vel[i, 0], self.agents_vel[i, 1]
            theta = self.agents_theta[i]
            s_self = [
                x / self.cfg.env_size, y / self.cfg.env_size,
                theta / np.pi, v / self.cfg.v_max, w / self.cfg.w_max
            ]

            # 2. 极简版局部地图 (以自身为中心截取 local_map_size)
            half_s = self.cfg.local_map_size // 2
            gx, gy = int(x / self.cfg.grid_res), int(y / self.cfg.grid_res)
            local_map = np.zeros((self.cfg.local_map_size, self.cfg.local_map_size))
            for dx in range(-half_s, half_s):
                for dy in range(-half_s, half_s):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.cfg.grid_size and 0 <= ny < self.cfg.grid_size:
                        local_map[dx + half_s, dy + half_s] = self.global_map[nx, ny]
                    else:
                        local_map[dx + half_s, dy + half_s] = -1  # 越界视为障碍物
            s_local = local_map.flatten().tolist()

            # 3. 雷达射线
            s_lidar = np.ones(self.cfg.num_lidar_rays).tolist()

            # 4. 队友共享信息
            s_shared = []
            for j in range(self.n):
                if i != j:
                    dx_r = (self.agents_pos[j, 0] - x) / self.cfg.env_size
                    dy_r = (self.agents_pos[j, 1] - y) / self.cfg.env_size
                    s_shared.extend([
                        dx_r, dy_r,
                        self.agents_vel[j, 0] / self.cfg.v_max,
                        self.agents_vel[j, 1] / self.cfg.w_max
                    ])

            # 5. 引导点特征（主引导点的距离和相对角度）
            s_guidance = self._get_guidance_feature(i)

            # 拼接
            agent_obs = np.concatenate([s_self, s_lidar, s_shared, s_local, s_guidance])
            obs.append(agent_obs)

        return np.array(obs, dtype=np.float32)

    @staticmethod
    def _distance_to_guidance(agent_pos: np.ndarray, guidance_world: np.ndarray) -> float:
        return float(np.hypot(agent_pos[0] - guidance_world[0], agent_pos[1] - guidance_world[1]))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _get_primary_guidance_world(self, guidance_points):
        if guidance_points is None or len(guidance_points) == 0:
            return None
        wx, wy = guidance_points[0]["world"]
        return np.array([wx, wy], dtype=np.float32)

    def _get_active_guidance_point(self):
        if not self.current_guidance_points:
            return None
        if self.current_guidance_idx < 0 or self.current_guidance_idx >= len(self.current_guidance_points):
            return None
        return self.current_guidance_points[self.current_guidance_idx]

    def _get_active_guidance_world(self):
        active_point = self._get_active_guidance_point()
        if active_point is None:
            return None
        wx, wy = active_point["world"]
        return np.array([wx, wy], dtype=np.float32)

    def _get_guidance_feature(self, agent_idx: int):
        guidance_world = self._get_active_guidance_world()
        if guidance_world is None:
            return [1.0, 0.0]

        x, y = self.agents_pos[agent_idx, 0], self.agents_pos[agent_idx, 1]
        theta = self.agents_theta[agent_idx]
        dx = float(guidance_world[0] - x)
        dy = float(guidance_world[1] - y)
        dist = float(np.hypot(dx, dy))
        dist_norm = dist / max(self.cfg.env_size, 1e-8)
        target_heading = float(np.arctan2(dy, dx))
        rel_angle = self._wrap_to_pi(target_heading - theta)
        rel_angle_norm = rel_angle / np.pi
        return [dist_norm, rel_angle_norm]

    def get_global_critic_obs(self):
        """
        构造 centralized critic 使用的全局状态:
        - map_obs: [C, H, W], 其中 C=3, H=W=grid_size
            ch0: 已覆盖区域(1/0)
            ch1: 障碍物区域(1/0)
            ch2: 智能体位置热图(单点置 1)
        - vec_obs: 低维全局向量
            [coverage_rate, step_ratio, agents_pos(flatten), agents_vel(flatten)]
        """
        gs = self.cfg.grid_size

        # 基于 global_map 构造二值语义图层
        covered_map = (self.global_map == 1).astype(np.float32)
        obstacle_map = (self.global_map == -1).astype(np.float32)

        # 智能体位置图层
        agent_map = np.zeros((gs, gs), dtype=np.float32)
        for i in range(self.n):
            x, y = self.agents_pos[i, 0], self.agents_pos[i, 1]
            gx = int(x / self.cfg.grid_res)
            gy = int(y / self.cfg.grid_res)
            gx = np.clip(gx, 0, gs - 1)
            gy = np.clip(gy, 0, gs - 1)
            agent_map[gx, gy] = 1.0

        map_obs = np.stack([covered_map, obstacle_map, agent_map], axis=0).astype(np.float32)

        total_valid_grids = np.sum(self.global_map >= 0)
        covered_grids = np.sum(self.global_map == 1)
        coverage_rate = covered_grids / max(total_valid_grids, 1)
        step_ratio = self.steps / max(self.cfg.max_steps, 1)

        pos_norm = (self.agents_pos / self.cfg.env_size).reshape(-1).astype(np.float32)
        vel_norm = self.agents_vel.copy()
        vel_norm[:, 0] = vel_norm[:, 0] / max(self.cfg.v_max, 1e-8)
        vel_norm[:, 1] = vel_norm[:, 1] / max(self.cfg.w_max, 1e-8)
        vel_norm = vel_norm.reshape(-1).astype(np.float32)

        vec_obs = np.concatenate([
            np.array([coverage_rate, step_ratio], dtype=np.float32),
            pos_norm,
            vel_norm
        ]).astype(np.float32)

        return map_obs, vec_obs


class Plot:
    """通用绘图类接口"""

    @staticmethod
    def plot_learning_curve(episode_rewards, title="Multi-UAV Coverage Learning Curve"):
        plt.figure(figsize=(10, 6))

        window_size = 50
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
            plt.plot(np.arange(len(moving_avg)) + window_size - 1, moving_avg, color='red', linewidth=2,
                     label='Moving Average (50 eps)')

        plt.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        plt.axhline(y=400, color='g', linestyle='--', label='Max Possible Return')
        plt.axhline(y=-400, color='r', linestyle='--', label='Min Possible Return')

        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()