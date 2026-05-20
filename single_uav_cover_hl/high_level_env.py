import os
from dataclasses import dataclass
from typing import List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from env import Config, MultiUAVCoverageEnv


@dataclass
class HighLevelConfig:
    # 上层每个 episode 允许的宏观决策步数（与下层微观 step 计数无关）；到时 truncated
    max_high_steps: int = 100
    # 上层每次决策后，下层最多执行多少步
    option_horizon: int = 20
    # 上层离散选区数量：将战场划分为 grid_bins x grid_bins
    grid_bins: int = 5
    # 判定“到达引导点”的距离阈值（米）
    reach_dist: float = 4.0

    # 上层奖励系数
    w_cov: float = 8.0
    w_new: float = 1.0
    w_travel: float = 0.2
    w_switch: float = 0.05
    w_fail: float = 2.0
    reach_bonus: float = 0.2


class HighLevelGuidanceEnv(gym.Env):
    """
    上层环境：
    - action: 选择候选引导点索引（Discrete）
    - 内部调用已训练的下层策略执行最多 option_horizon 个微观 step（仅占位执行 option）
    - episode 时长由上层 max_high_steps 约束；不因下层微观 max_steps 而结束整条上层回合（下层微观预算在封装时拉大）
    - reward: 由覆盖增量、代价、失败惩罚构成
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        low_level_model_path: str,
        env_cfg: Optional[Config] = None,
        hl_cfg: Optional[HighLevelConfig] = None,
        deterministic_low_level: bool = True,
        infer_mode: Optional[bool] = None,
    ):
        super().__init__()
        self.hl_cfg = hl_cfg if hl_cfg is not None else HighLevelConfig()
        self.env_cfg = env_cfg if env_cfg is not None else Config()
        self.env_cfg.num_agents = 1
        if infer_mode is not None:
            self.env_cfg.infer_mode = bool(infer_mode)

        # 保证同一上层 episode 内：累计微观步不会超过 max_steps，避免下层因步数上限截断从而打断连贯覆盖过程
        _macro = max(int(self.hl_cfg.max_high_steps), 1)
        _opt = max(int(self.hl_cfg.option_horizon), 1)
        _need_micro_budget = _macro * _opt + max(int(self.env_cfg.max_steps), 1)
        self.env_cfg.max_steps = max(int(self.env_cfg.max_steps), _need_micro_budget)

        self.low_env = MultiUAVCoverageEnv(self.env_cfg)
        self.low_level_model_path = low_level_model_path
        if not os.path.exists(self.low_level_model_path):
            raise FileNotFoundError(f"low-level model not found: {self.low_level_model_path}")
        self.low_level_model = PPO.load(self.low_level_model_path, device="cpu")
        self.deterministic_low_level = deterministic_low_level

        self.num_regions = int(self.hl_cfg.grid_bins * self.hl_cfg.grid_bins)
        self.action_space = spaces.Discrete(self.num_regions)
        # 观测: 基础(coverage, step_ratio, x, y, theta)
        # + 每个区域(距离, 角度, 未覆盖占比, 已扫描小栅格数量)
        obs_dim = 5 + self.num_regions * 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.last_action: Optional[int] = None
        # 设为 True 时，每个下层 step 后把低层状态追加到 low_frame_buffer（供可视化 GIF）
        self.record_low_frames = False
        self.low_frame_buffer: List[dict] = []
        self.high_macro_step: int = 0

    def _append_low_frame(self):
        self.low_frame_buffer.append(
            {
                "pos": self.low_env.agents_pos.copy(),
                "theta": self.low_env.agents_theta.copy(),
                "map": self.low_env.global_map.copy(),
                "guidance_points": list(self.low_env.current_guidance_points)
                if self.low_env.current_guidance_points
                else [],
            }
        )

    def reset(self, seed=None, options=None):
        self.low_env.reset(seed=seed)
        self.high_macro_step = 0
        self.last_action = None
        self.low_frame_buffer.clear()
        if self.record_low_frames:
            self._append_low_frame()
        return self._get_high_level_obs(), {}

    def step(self, action):
        action = int(action)
        cov_before, covered_before = self._coverage_stats()
        pos_before = self.low_env.agents_pos[0].copy()

        invalid_action = action < 0 or action >= self.num_regions
        target_reached = False
        collision_or_oob = False
        terminated = False
        low_trunc_micro = False  # 下层微观 trunc：只结束本轮 option，不结束上层 episode
        option_end_reason = "horizon"

        if invalid_action:
            # 动作无效时退化到第一个区域
            action = 0

        target_world = self._action_to_world(action)
        # option 起点到本步引导点的距离；已在半径内则不应再拿 reach_bonus（防重复刷区站桩）
        dist_before_target = float(np.linalg.norm(pos_before - target_world))
        # 将上层选择的区域中心坐标写入低层目标（单目标）
        self.low_env.current_guidance_points = [
            {
                "grid": (
                    int(np.clip(target_world[0] / self.low_env.cfg.grid_res, 0, self.low_env.cfg.grid_size - 1)),
                    int(np.clip(target_world[1] / self.low_env.cfg.grid_res, 0, self.low_env.cfg.grid_size - 1)),
                ),
                "world": (float(target_world[0]), float(target_world[1])),
                "score": 0.0,
            }
        ]
        self.low_env.current_guidance_idx = 0

        executed_steps = 0
        for _ in range(self.hl_cfg.option_horizon):
            low_obs = self.low_env._get_obs()[0]
            low_action, _ = self.low_level_model.predict(low_obs, deterministic=self.deterministic_low_level)
            _, _, done_all, trunc, _ = self.low_env.step(np.array([low_action], dtype=np.float32))
            executed_steps += 1
            if self.record_low_frames:
                self._append_low_frame()

            dist = float(np.linalg.norm(self.low_env.agents_pos[0] - target_world))
            if dist < self.hl_cfg.reach_dist:
                target_reached = True
                option_end_reason = "reach_dist"
                break

            if done_all:
                terminated = True
                # done_all 来源：碰撞越界 或 覆盖达标
                x, y = self.low_env.agents_pos[0]
                if x < 0 or x > self.env_cfg.env_size or y < 0 or y > self.env_cfg.env_size:
                    collision_or_oob = True
                for ox, oy, orad in self.low_env.obstacles:
                    if np.hypot(x - ox, y - oy) < (self.env_cfg.uav_radius + orad):
                        collision_or_oob = True
                        break
                option_end_reason = "low_collision_oob" if collision_or_oob else "low_done_coverage"
                break
            # if trunc:
            #     low_trunc_micro = True
            #     option_end_reason = "low_micro_truncated"
            #     break

        cov_after, covered_after = self._coverage_stats()
        pos_after = self.low_env.agents_pos[0].copy()

        delta_cov = cov_after - cov_before
        delta_new = (covered_after - covered_before) / max(np.sum(self.low_env.global_map >= 0), 1)
        travel_dist = float(np.linalg.norm(pos_after - pos_before))
        switch_cost = 1.0 if (self.last_action is not None and self.last_action != action) else 0.0
        fail_cost = 1.0 if collision_or_oob else 0.0

        reward = (
            self.hl_cfg.w_cov * delta_cov
            + self.hl_cfg.w_new * delta_new
            - self.hl_cfg.w_travel * (travel_dist / max(self.env_cfg.env_size, 1e-8))
            - self.hl_cfg.w_switch * switch_cost
            - self.hl_cfg.w_fail * fail_cost
            - 0.05
        )
        reached_from_outside = target_reached and dist_before_target >= self.hl_cfg.reach_dist - 1e-6
        if reached_from_outside:
            reward += self.hl_cfg.reach_bonus
        if invalid_action:
            reward -= 0.2

        # 覆盖率达到下层配置的 coverage_goal（且无碰撞/OOB）：本层 episode 成功终止并给予与下层一致的达标奖励
        coverage_success = (cov_after >= self.env_cfg.coverage_goal) and not collision_or_oob
        if coverage_success:
            reward += float(self.env_cfg.r_cov_goal)
            terminated = True
            if option_end_reason == "horizon":
                option_end_reason = "coverage_goal_met"

        self.high_macro_step += 1
        truncated = (
            self.high_macro_step >= self.hl_cfg.max_high_steps and not terminated
        )

        self.last_action = action
        obs = self._get_high_level_obs()

        info = {
            "delta_cov": float(delta_cov),
            "delta_new": float(delta_new),
            "travel_dist": float(travel_dist),
            "executed_steps": int(executed_steps),
            "target_reached": bool(target_reached),
            "reach_bonus_applied": bool(reached_from_outside),
            "invalid_action": bool(invalid_action),
            "num_regions": int(self.num_regions),
            "coverage_success": bool(coverage_success),
            "coverage_rate": float(cov_after),
            "option_end_reason": option_end_reason,
            "high_macro_step": int(self.high_macro_step),
            "low_trunc_micro": bool(low_trunc_micro),
        }

        if self.env_cfg.infer_mode and (terminated or truncated):
            if coverage_success:
                episode_end_reason = "coverage_goal_met"
            elif collision_or_oob:
                episode_end_reason = "collision_oob"
            elif truncated:
                episode_end_reason = (
                    f"high_macro_steps_exhausted(>={self.hl_cfg.max_high_steps})"
                )
            else:
                episode_end_reason = "terminated_other"
            print(
                f"[HighLevel infer] Episode结束: 原因={episode_end_reason}, "
                f"上层步={self.high_macro_step}/{self.hl_cfg.max_high_steps}, "
                f"最后一跳option={option_end_reason}, 覆盖率={cov_after:.4f}"
            )

        return obs, float(reward), terminated, truncated, info

    def _coverage_stats(self):
        total_valid = np.sum(self.low_env.global_map >= 0)
        covered = np.sum(self.low_env.global_map == 1)
        return float(covered / max(total_valid, 1)), int(covered)

    def _action_to_world(self, action_idx: int):
        bins = self.hl_cfg.grid_bins
        cell_size = self.env_cfg.env_size / float(bins)
        gx = int(action_idx // bins)
        gy = int(action_idx % bins)
        wx = (gx + 0.5) * cell_size
        wy = (gy + 0.5) * cell_size
        return np.array([wx, wy], dtype=np.float32)

    def _region_grid_slice(self, action_idx: int):
        bins = self.hl_cfg.grid_bins
        gs = self.low_env.cfg.grid_size
        cell = gs / float(bins)
        gx = int(action_idx // bins)
        gy = int(action_idx % bins)
        x0 = int(np.floor(gx * cell))
        x1 = int(np.floor((gx + 1) * cell))
        y0 = int(np.floor(gy * cell))
        y1 = int(np.floor((gy + 1) * cell))
        x0 = int(np.clip(x0, 0, gs))
        x1 = int(np.clip(x1, x0 + 1, gs))
        y0 = int(np.clip(y0, 0, gs))
        y1 = int(np.clip(y1, y0 + 1, gs))
        return x0, x1, y0, y1

    def _region_unknown_ratio(self, action_idx: int):
        x0, x1, y0, y1 = self._region_grid_slice(action_idx)

        region = self.low_env.global_map[x0:x1, y0:y1]
        valid = (region >= 0)
        valid_count = int(np.sum(valid))
        if valid_count <= 0:
            return 0.0
        unknown_count = int(np.sum(region == 0))
        return float(unknown_count / valid_count)

    def _region_scanned_count(self, action_idx: int):
        x0, x1, y0, y1 = self._region_grid_slice(action_idx)
        region = self.low_env.global_map[x0:x1, y0:y1]
        return float(np.sum(region == 1))

    def _get_high_level_obs(self):
        cov_rate, _ = self._coverage_stats()
        step_ratio = self.high_macro_step / max(self.hl_cfg.max_high_steps, 1)
        x, y = self.low_env.agents_pos[0]
        theta = self.low_env.agents_theta[0]

        base = [
            cov_rate,
            step_ratio,
            x / self.env_cfg.env_size,
            y / self.env_cfg.env_size,
            float(theta / np.pi),
        ]

        cand_feat = []
        for i in range(self.num_regions):
            wx, wy = self._action_to_world(i)
            dx, dy = wx - x, wy - y
            dist_norm = float(np.hypot(dx, dy) / max(self.env_cfg.env_size, 1e-8))
            rel_angle = float((np.arctan2(dy, dx) - theta + np.pi) % (2 * np.pi) - np.pi) / np.pi
            unknown_ratio = self._region_unknown_ratio(i)
            scanned_count = self._region_scanned_count(i)
            cand_feat.extend([dist_norm, rel_angle, unknown_ratio, scanned_count])

        return np.array(base + cand_feat, dtype=np.float32)
