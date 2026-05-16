"""
多智能体集中式高层环境（阶段一）

架构：
  集中式 PPO 上层 (MultiDiscrete action)
    -> 每个 agent 分配一个目标区域
    -> 共享下层 PPO 模型执行 option_horizon 步微观控制
    -> 返回全局 team reward

动作空间: MultiDiscrete([num_regions] * num_agents)
观测空间: 全局覆盖率 + step_ratio + 每个 agent 状态 + 每个区域特征 + agent-region 配对特征
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_THIS_DIR, "simulation", ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

# Checkpoints saved on NumPy 2 expect `numpy._core.numeric`. NumPy 1.26 may
# not expose that import path, so register a compatibility alias before PPO.load.
if int(np.__version__.split(".", maxsplit=1)[0]) < 2:
    import importlib.util

    if importlib.util.find_spec("numpy._core.numeric") is None:
        import numpy.core.numeric as _np_core_numeric

        _np_core = sys.modules.get("numpy._core")
        if _np_core is None:
            _np_core = types.ModuleType("numpy._core")
            sys.modules["numpy._core"] = _np_core
        setattr(_np_core, "numeric", _np_core_numeric)
        sys.modules["numpy._core.numeric"] = _np_core_numeric

from env import Config, MultiUAVCoverageEnv


@dataclass
class MultiHighLevelConfig:
    # 上层每个 episode 允许的宏观决策步数
    max_high_steps: int = 80
    # 上层每次决策后，下层最多执行多少步
    option_horizon: int = 15
    # 地图划分粒度：grid_bins x grid_bins 个区域
    grid_bins: int = 5
    # 判定"到达引导点"的距离阈值（米）：用于 reach_bonus 统计
    reach_dist: float = 4.0
    # 任意 agent 到达引导点时提前结束当前 option 的距离阈值（米）
    early_reach_dist: float = 4.0

    # 奖励系数
    w_cov: float = 8.0       # 全局覆盖率增量
    w_new: float = 1.0       # 新覆盖栅格增量
    w_travel: float = 0.1    # 移动距离惩罚（按 agent 平均）
    w_fail: float = 2.0      # 碰撞/越界惩罚
    w_duplicate: float = 0.3 # 多 agent 选同一区域惩罚
    w_target_near: float = 0.1 # 多 agent 目标区域过近惩罚
    w_switch: float = 0.03    # agent 切换目标区域惩罚
    w_scan_overlap: float = 0.4 # 多 agent 局部扫描重叠惩罚
    w_agent_proximity: float = 0.2 # agent 之间距离过近惩罚
    w_agent_collision: float = 2.0 # agent-agent 碰撞惩罚
    reach_bonus: float = 0.15 # 到达引导点奖励（每个 agent 独立计算）
    time_penalty: float = 0.03 # 每步时间惩罚
    w_fully_scanned: float = 0.5 # 目标区域已被完全扫描时的惩罚（每个 agent）
    min_target_dist: float = 6.0 # 两个目标点小于该距离时视为分配过近
    min_agent_dist: float = 2.0 # agent 间低于该距离时给 proximity penalty


class MultiAgentHighLevelEnv(gym.Env):
    """
    集中式多智能体高层环境。

    下层 env 的 current_guidance_points / current_guidance_idx 仍保留，
    但本环境额外维护 agent_guidance_points dict，并在每次 option 执行前
    将对应 agent 的引导点写入下层 env，使下层 _get_guidance_feature 能
    正确读取各自的目标。

    由于下层 _get_guidance_feature 读取的是 current_guidance_points[0]，
    我们在每个 agent 的 predict 前临时切换引导点，predict 后恢复。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        low_level_model_path: str,
        env_cfg: Optional[Config] = None,
        hl_cfg: Optional[MultiHighLevelConfig] = None,
        num_agents: int = 3,
        deterministic_low_level: bool = True,
        infer_mode: Optional[bool] = None,
    ):
        super().__init__()
        self.hl_cfg = hl_cfg if hl_cfg is not None else MultiHighLevelConfig()
        self.env_cfg = env_cfg if env_cfg is not None else Config()
        self.num_agents = num_agents
        self.env_cfg.num_agents = num_agents
        if infer_mode is not None:
            self.env_cfg.infer_mode = bool(infer_mode)

        # 保证下层微观步数预算足够，不会因 max_steps 截断上层 episode
        _macro = max(int(self.hl_cfg.max_high_steps), 1)
        _opt = max(int(self.hl_cfg.option_horizon), 1)
        _need = _macro * _opt + max(int(self.env_cfg.max_steps), 1)
        self.env_cfg.max_steps = max(int(self.env_cfg.max_steps), _need)

        self.low_env = MultiUAVCoverageEnv(self.env_cfg)

        if not os.path.exists(low_level_model_path):
            raise FileNotFoundError(f"low-level model not found: {low_level_model_path}")
        self.low_level_model = PPO.load(low_level_model_path, device="cpu")
        self.deterministic_low_level = deterministic_low_level

        self.num_regions = self.hl_cfg.grid_bins ** 2

        # 动作空间：每个 agent 独立选一个区域
        self.action_space = spaces.MultiDiscrete([self.num_regions] * self.num_agents)

        # 观测空间维度：
        #   全局: coverage_rate(1) + step_ratio(1) = 2
        #   每个 agent: x, y, theta, v, w, nearest_agent_dist, last_region(7) = 7 * num_agents
        #   每个区域: unknown_ratio, scanned_count(2) = 2 * num_regions
        #   每个 (agent, region) 对: dist_norm, rel_angle(2) = 2 * num_agents * num_regions
        obs_dim = (
            2
            + 7 * self.num_agents
            + 2 * self.num_regions
            + 2 * self.num_agents * self.num_regions
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 每个 agent 当前分配的引导点（world 坐标）
        self.agent_guidance: Dict[int, Optional[np.ndarray]] = {i: None for i in range(self.num_agents)}
        self.last_actions: Optional[np.ndarray] = None
        self.high_macro_step: int = 0

        # 可视化帧缓冲
        self.record_low_frames: bool = False
        self.low_frame_buffer: List[dict] = []

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self.low_env.reset(seed=seed)
        self.high_macro_step = 0
        self.last_actions = None
        self.agent_guidance = {i: None for i in range(self.num_agents)}
        self.low_frame_buffer.clear()
        if self.record_low_frames:
            self._append_low_frame()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.int32).reshape(-1)
        if action.shape[0] != self.num_agents:
            raise ValueError(f"expected action shape ({self.num_agents},), got {action.shape}")
        selected_regions = np.clip(action, 0, self.num_regions - 1).astype(np.int32)

        cov_before, covered_before = self._coverage_stats()
        pos_before = self.low_env.agents_pos.copy()  # (n, 2)

        # 将每个 agent 的动作转成世界坐标引导点
        target_worlds: List[np.ndarray] = []
        for i in range(self.num_agents):
            region_id = int(selected_regions[i])
            tw = self._action_to_world(region_id)
            target_worlds.append(tw)
            self.agent_guidance[i] = tw

        # 目标冲突惩罚：多个 agent 选同一区域
        duplicate_count = self.num_agents - len(set(int(a) for a in selected_regions))
        duplicate_penalty = duplicate_count * self.hl_cfg.w_duplicate
        target_near_penalty = self._target_near_penalty(target_worlds)
        switch_count = self._switch_count(selected_regions)
        switch_penalty = switch_count * self.hl_cfg.w_switch
        fully_scanned_count = sum(
            1 for i in range(self.num_agents)
            if self._region_fully_scanned(int(selected_regions[i]))
        )
        fully_scanned_penalty = fully_scanned_count * self.hl_cfg.w_fully_scanned

        # 执行 option_horizon 步微观控制
        terminated = False
        collision_or_oob = False
        low_truncated = False
        agents_reached = [False] * self.num_agents

        for _ in range(self.hl_cfg.option_horizon):
            # 为每个 agent 分别预测动作（临时切换引导点）
            # 下层模型用 num_agents=1 训练，需要单 agent 观测（无队友特征）
            low_actions = []
            for i in range(self.num_agents):
                self._set_guidance_for_agent(i)
                low_obs_i = self._get_single_agent_obs(i)
                low_action_i, _ = self.low_level_model.predict(
                    low_obs_i, deterministic=self.deterministic_low_level
                )
                low_actions.append(low_action_i)

            _, _, done_all, trunc, _ = self.low_env.step(
                np.array(low_actions, dtype=np.float32)
            )
            if self.record_low_frames:
                self._append_low_frame()
            if trunc:
                low_truncated = True

            # 检查各 agent 是否到达目标
            any_reached = False
            for i in range(self.num_agents):
                dist = float(np.linalg.norm(self.low_env.agents_pos[i] - target_worlds[i]))
                if dist < self.hl_cfg.reach_dist:
                    agents_reached[i] = True
                if dist < self.hl_cfg.early_reach_dist:
                    any_reached = True

            if done_all:
                terminated = True
                collision_or_oob = self._check_collision_oob()
                break

            # 任意 agent 到达引导点时提前结束当前 option，触发新的高层决策
            if any_reached:
                break

        cov_after, covered_after = self._coverage_stats()
        pos_after = self.low_env.agents_pos.copy()

        # 计算 team reward
        delta_cov = cov_after - cov_before
        delta_new = (covered_after - covered_before) / max(
            int(np.sum(self.low_env.global_map >= 0)), 1
        )
        avg_travel = float(
            np.mean([np.linalg.norm(pos_after[i] - pos_before[i]) for i in range(self.num_agents)])
        )
        scan_overlap_ratio = self._scan_overlap_ratio()
        agent_proximity_penalty = self._agent_proximity_penalty()
        agent_collision = self._check_agent_collision()
        if agent_collision:
            collision_or_oob = True
        reach_bonus = sum(
            self.hl_cfg.reach_bonus
            for i, reached in enumerate(agents_reached)
            if reached and float(np.linalg.norm(pos_before[i] - target_worlds[i])) >= self.hl_cfg.reach_dist - 1e-6
        )

        reward = (
            self.hl_cfg.w_cov * delta_cov
            + self.hl_cfg.w_new * delta_new
            - self.hl_cfg.w_travel * (avg_travel / max(self.env_cfg.env_size, 1e-8))
            - self.hl_cfg.w_fail * float(collision_or_oob)
            - duplicate_penalty
            - self.hl_cfg.w_target_near * target_near_penalty
            - switch_penalty
            - self.hl_cfg.w_scan_overlap * scan_overlap_ratio
            - self.hl_cfg.w_agent_proximity * agent_proximity_penalty
            - self.hl_cfg.w_agent_collision * float(agent_collision)
            - self.hl_cfg.time_penalty
            + reach_bonus
            - fully_scanned_penalty
        )

        coverage_success = (cov_after >= self.env_cfg.coverage_goal) and not collision_or_oob
        if coverage_success:
            reward += float(self.env_cfg.r_cov_goal)
            terminated = True

        self.high_macro_step += 1
        truncated = self.high_macro_step >= self.hl_cfg.max_high_steps and not terminated
        self.last_actions = selected_regions.copy()

        info = {
            "coverage_rate": float(cov_after),
            "delta_cov": float(delta_cov),
            "delta_new": float(delta_new),
            "avg_travel": float(avg_travel),
            "duplicate_count": int(duplicate_count),
            "target_near_penalty": float(target_near_penalty),
            "switch_count": int(switch_count),
            "scan_overlap_ratio": float(scan_overlap_ratio),
            "fully_scanned_count": int(fully_scanned_count),
            "agent_proximity_penalty": float(agent_proximity_penalty),
            "agent_collision": bool(agent_collision),
            "collision_or_oob": bool(collision_or_oob),
            "coverage_success": bool(coverage_success),
            "agents_reached": agents_reached,
            "high_macro_step": int(self.high_macro_step),
            "low_truncated": bool(low_truncated),
            "selected_regions": selected_regions.astype(int).tolist(),
        }

        if self.env_cfg.infer_mode and (terminated or truncated):
            reason = (
                "coverage_goal" if coverage_success
                else "collision_oob" if collision_or_oob
                else f"max_steps({self.hl_cfg.max_high_steps})"
            )
            print(
                f"[MultiHL] episode end: {reason}, "
                f"step={self.high_macro_step}, cov={cov_after:.4f}, "
                f"dup={duplicate_count}, overlap={scan_overlap_ratio:.3f}"
            )

        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        cov_rate, _ = self._coverage_stats()
        step_ratio = self.high_macro_step / max(self.hl_cfg.max_high_steps, 1)

        global_feat = [cov_rate, step_ratio]

        # 每个 agent 的位置、速度、最近邻距离和上一轮区域
        agent_feat = []
        for i in range(self.num_agents):
            x, y = self.low_env.agents_pos[i]
            theta = self.low_env.agents_theta[i]
            v, w = self.low_env.agents_vel[i]
            nearest_dist = self._nearest_agent_dist(i)
            last_region = -1.0
            if self.last_actions is not None:
                last_region = float(self.last_actions[i] / max(self.num_regions - 1, 1))
            agent_feat.extend([
                x / self.env_cfg.env_size,
                y / self.env_cfg.env_size,
                float(theta / np.pi),
                float(v / max(self.env_cfg.v_max, 1e-8)),
                float(w / max(self.env_cfg.w_max, 1e-8)),
                float(nearest_dist / max(self.env_cfg.env_size, 1e-8)),
                last_region,
            ])

        # 每个区域的全局特征
        region_feat = []
        for r in range(self.num_regions):
            region_feat.extend([
                self._region_unknown_ratio(r),
                self._region_scanned_count(r) / max(self._region_total_valid(r), 1),
            ])

        # 每个 (agent, region) 对的距离和相对角度
        pair_feat = []
        for i in range(self.num_agents):
            x, y = self.low_env.agents_pos[i]
            theta = self.low_env.agents_theta[i]
            for r in range(self.num_regions):
                wx, wy = self._action_to_world(r)
                dx, dy = wx - x, wy - y
                dist_norm = float(np.hypot(dx, dy) / max(self.env_cfg.env_size, 1e-8))
                rel_angle = float(
                    (np.arctan2(dy, dx) - theta + np.pi) % (2 * np.pi) - np.pi
                ) / np.pi
                pair_feat.extend([dist_norm, rel_angle])

        return np.array(global_feat + agent_feat + region_feat + pair_feat, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_single_agent_obs(self, agent_idx: int) -> np.ndarray:
        """
        为 agent_idx 构造与下层单 agent 模型兼容的观测（num_agents=1，无队友特征）。
        临时把 low_env.n 设为 1，只取该 agent 的状态行，然后恢复。
        """
        saved_n = self.low_env.n
        saved_cfg_n = self.low_env.cfg.num_agents

        saved_pos = self.low_env.agents_pos.copy()
        saved_theta = self.low_env.agents_theta.copy()
        saved_vel = self.low_env.agents_vel.copy()

        try:
            # 临时把 agent_idx 的状态移到 index 0，使观测维度与单 agent 低层模型一致。
            self.low_env.agents_pos = self.low_env.agents_pos[agent_idx:agent_idx + 1].copy()
            self.low_env.agents_theta = self.low_env.agents_theta[agent_idx:agent_idx + 1].copy()
            self.low_env.agents_vel = self.low_env.agents_vel[agent_idx:agent_idx + 1].copy()
            self.low_env.n = 1
            self.low_env.cfg.num_agents = 1
            obs = self.low_env._get_obs()[0]  # shape (single_obs_dim,)
        finally:
            self.low_env.agents_pos = saved_pos
            self.low_env.agents_theta = saved_theta
            self.low_env.agents_vel = saved_vel
            self.low_env.n = saved_n
            self.low_env.cfg.num_agents = saved_cfg_n

        return obs

    def _set_guidance_for_agent(self, agent_idx: int):
        """将 agent_idx 的引导点写入下层 env，供 _get_guidance_feature 读取。"""
        gw = self.agent_guidance[agent_idx]
        if gw is None:
            self.low_env.current_guidance_points = []
            self.low_env.current_guidance_idx = 0
            return
        self.low_env.current_guidance_points = [
            {
                "grid": (
                    int(np.clip(gw[0] / self.low_env.cfg.grid_res, 0, self.low_env.cfg.grid_size - 1)),
                    int(np.clip(gw[1] / self.low_env.cfg.grid_res, 0, self.low_env.cfg.grid_size - 1)),
                ),
                "world": (float(gw[0]), float(gw[1])),
                "score": 0.0,
            }
        ]
        self.low_env.current_guidance_idx = 0

    def _action_to_world(self, action_idx: int) -> np.ndarray:
        bins = self.hl_cfg.grid_bins
        cell_size = self.env_cfg.env_size / float(bins)
        gx = int(action_idx // bins)
        gy = int(action_idx % bins)
        return np.array([(gx + 0.5) * cell_size, (gy + 0.5) * cell_size], dtype=np.float32)

    def _region_grid_slice(self, action_idx: int) -> Tuple[int, int, int, int]:
        bins = self.hl_cfg.grid_bins
        gs = self.low_env.cfg.grid_size
        cell = gs / float(bins)
        gx = int(action_idx // bins)
        gy = int(action_idx % bins)
        x0 = int(np.clip(int(np.floor(gx * cell)), 0, gs))
        x1 = int(np.clip(int(np.floor((gx + 1) * cell)), x0 + 1, gs))
        y0 = int(np.clip(int(np.floor(gy * cell)), 0, gs))
        y1 = int(np.clip(int(np.floor((gy + 1) * cell)), y0 + 1, gs))
        return x0, x1, y0, y1

    def _region_unknown_ratio(self, action_idx: int) -> float:
        x0, x1, y0, y1 = self._region_grid_slice(action_idx)
        region = self.low_env.global_map[x0:x1, y0:y1]
        valid = int(np.sum(region >= 0))
        if valid <= 0:
            return 0.0
        return float(np.sum(region == 0)) / valid

    def _region_scanned_count(self, action_idx: int) -> float:
        x0, x1, y0, y1 = self._region_grid_slice(action_idx)
        return float(np.sum(self.low_env.global_map[x0:x1, y0:y1] == 1))

    def _region_total_valid(self, action_idx: int) -> int:
        x0, x1, y0, y1 = self._region_grid_slice(action_idx)
        return int(np.sum(self.low_env.global_map[x0:x1, y0:y1] >= 0))

    def _region_fully_scanned(self, action_idx: int) -> bool:
        total = self._region_total_valid(action_idx)
        if total <= 0:
            return False
        return self._region_unknown_ratio(action_idx) == 0.0

    def _coverage_stats(self) -> Tuple[float, int]:
        total_valid = int(np.sum(self.low_env.global_map >= 0))
        covered = int(np.sum(self.low_env.global_map == 1))
        return float(covered / max(total_valid, 1)), covered

    def _switch_count(self, selected_regions: np.ndarray) -> int:
        if self.last_actions is None:
            return 0
        return int(np.sum(np.asarray(self.last_actions, dtype=np.int32) != selected_regions))

    def _target_near_penalty(self, target_worlds: List[np.ndarray]) -> float:
        if self.num_agents <= 1:
            return 0.0
        penalty = 0.0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = float(np.linalg.norm(target_worlds[i] - target_worlds[j]))
                if dist < self.hl_cfg.min_target_dist:
                    penalty += (self.hl_cfg.min_target_dist - dist) / max(self.hl_cfg.min_target_dist, 1e-8)
        return float(penalty)

    def _nearest_agent_dist(self, agent_idx: int) -> float:
        if self.num_agents <= 1:
            return float(self.env_cfg.env_size)
        pos = self.low_env.agents_pos[agent_idx]
        dists = [
            float(np.linalg.norm(pos - self.low_env.agents_pos[j]))
            for j in range(self.num_agents)
            if j != agent_idx
        ]
        return min(dists) if dists else float(self.env_cfg.env_size)

    def _agent_proximity_penalty(self) -> float:
        if self.num_agents <= 1:
            return 0.0
        penalty = 0.0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = float(np.linalg.norm(self.low_env.agents_pos[i] - self.low_env.agents_pos[j]))
                if dist < self.hl_cfg.min_agent_dist:
                    penalty += (self.hl_cfg.min_agent_dist - dist) / max(self.hl_cfg.min_agent_dist, 1e-8)
        return float(penalty)

    def _check_agent_collision(self) -> bool:
        if self.num_agents <= 1:
            return False
        min_dist = 2.0 * float(self.env_cfg.uav_radius)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.linalg.norm(self.low_env.agents_pos[i] - self.low_env.agents_pos[j]) < min_dist:
                    return True
        return False

    def _scan_overlap_ratio(self) -> float:
        """Approximate current detector-circle overlap with grid counts."""
        gs = self.low_env.cfg.grid_size
        counts = np.zeros((gs, gs), dtype=np.int16)
        grad = int(self.low_env.cfg.det_radius / self.low_env.cfg.grid_res)
        for i in range(self.num_agents):
            x, y = self.low_env.agents_pos[i]
            gx = int(x / self.low_env.cfg.grid_res)
            gy = int(y / self.low_env.cfg.grid_res)
            for dx in range(-grad, grad + 1):
                for dy in range(-grad, grad + 1):
                    if dx * dx + dy * dy > grad * grad:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < gs and 0 <= ny < gs and self.low_env.global_map[nx, ny] >= 0:
                        counts[nx, ny] += 1
        overlap_cells = int(np.sum(counts > 1))
        scanned_cells = int(np.sum(counts > 0))
        return float(overlap_cells / max(scanned_cells, 1))

    def _check_collision_oob(self) -> bool:
        for i in range(self.num_agents):
            x, y = self.low_env.agents_pos[i]
            if x < 0 or x > self.env_cfg.env_size or y < 0 or y > self.env_cfg.env_size:
                return True
            for ox, oy, orad in self.low_env.obstacles:
                if np.hypot(x - ox, y - oy) < (self.env_cfg.uav_radius + orad):
                    return True
        return self._check_agent_collision()

    def _append_low_frame(self):
        self.low_frame_buffer.append({
            "pos": self.low_env.agents_pos.copy(),
            "theta": self.low_env.agents_theta.copy(),
            "map": self.low_env.global_map.copy(),
            "agent_guidance": {i: (gw.copy() if gw is not None else None)
                               for i, gw in self.agent_guidance.items()},
        })
