"""
Run the trained multi-agent high-level policy and render a 3D simulation.

Each UAV is shown at a fixed altitude with its own trajectory, detection circle,
and guidance point.  The coverage map is shared across all agents.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(SIM_DIR / ".mplconfig"))
(SIM_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

# libstdc++ fix: prefer conda's newer libstdc++ over the system one
_conda_lib = os.path.join(sys.prefix, "lib")
_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
if _conda_lib not in _ld_path:
    os.environ["LD_LIBRARY_PATH"] = f"{_conda_lib}:{_ld_path}"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# stable_baseline3/ 根目录（用于 Cover_multi.xxx 包导入）
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Cover_multi/ 目录（用于裸 import env / multi_high_level_env）
_COVER_MULTI = str(REPO_ROOT / "Cover_multi")
if _COVER_MULTI not in sys.path:
    sys.path.insert(0, _COVER_MULTI)

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from stable_baselines3 import PPO

from env import Config
from multi_high_level_env import MultiAgentHighLevelEnv, MultiHighLevelConfig


# ── colour palette ────────────────────────────────────────────────────────────
COVERED_COLOR   = "#bfdbfe"
UNKNOWN_COLOR   = "#f8fafc"
OBSTACLE_COLOR  = "#4b5563"
DETECTION_COLOR = "#1e40af"
GUIDANCE_COLOR  = "#dc2626"
UAV_BODY_COLOR  = "#2563eb"
UAV_ARM_COLOR   = "#111827"
START_COLOR     = "#1d4ed8"
GROUND_Z        = -0.04

# Per-agent accent colours (body / trajectory / guidance)
AGENT_COLORS = [
    ("#2563eb", "#f97316", "#dc2626"),   # agent 0: blue body, orange traj, red guidance
    ("#16a34a", "#a21caf", "#15803d"),   # agent 1: green body, purple traj, dark-green guidance
    ("#b45309", "#0891b2", "#92400e"),   # agent 2: amber body, cyan traj, dark-amber guidance
    ("#7c3aed", "#e11d48", "#5b21b6"),   # agent 3
    ("#0f766e", "#f59e0b", "#134e4a"),   # agent 4
]

GROUND_ZORDER    = 0
SCAN_ZORDER      = 18
TRAJECTORY_ZORDER = 24
GUIDANCE_ZORDER  = 30
UAV_ZORDER       = 40


# ── data classes ──────────────────────────────────────────────────────────────
@dataclass
class RolloutResult:
    frames: list[dict]
    high_steps: int
    total_reward: float
    coverage_rate: float
    end_reason: str


# ── helpers ───────────────────────────────────────────────────────────────────
def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _agent_colors(agent_idx: int) -> tuple[str, str, str]:
    return AGENT_COLORS[agent_idx % len(AGENT_COLORS)]


def build_env(
    low_model: Path,
    num_agents: int,
    grid_bins: int,
    option_horizon: int,
    max_high_steps: int,
) -> MultiAgentHighLevelEnv:
    env_cfg = Config()
    env_cfg.coverage_goal = 0.98
    env_cfg.infer_mode = False

    hl_cfg = MultiHighLevelConfig(
        grid_bins=grid_bins,
        option_horizon=option_horizon,
        max_high_steps=max_high_steps,
    )

    return MultiAgentHighLevelEnv(
        low_level_model_path=str(low_model),
        env_cfg=env_cfg,
        hl_cfg=hl_cfg,
        num_agents=num_agents,
        deterministic_low_level=True,
        infer_mode=False,
    )


def run_policy_rollout(
    env: MultiAgentHighLevelEnv,
    high_model: PPO,
    seed: int,
) -> RolloutResult:
    env.record_low_frames = True
    env.low_frame_buffer.clear()

    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    high_steps = 0
    total_reward = 0.0
    latest_info: dict = {}
    safety_cap = max(int(env.hl_cfg.max_high_steps), 1) + 32

    while not (terminated or truncated) and high_steps < safety_cap:
        action, _ = high_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, latest_info = env.step(action)
        total_reward += float(reward)
        high_steps += 1

    env.record_low_frames = False

    coverage_rate = float(latest_info.get("coverage_rate", env._coverage_stats()[0]))
    if latest_info.get("coverage_success"):
        end_reason = "coverage_goal_met"
    elif truncated:
        end_reason = "high_level_truncated"
    elif terminated:
        end_reason = "collision_oob" if latest_info.get("collision_or_oob") else "terminated"
    else:
        end_reason = "safety_cap"

    return RolloutResult(
        frames=list(env.low_frame_buffer),
        high_steps=high_steps,
        total_reward=total_reward,
        coverage_rate=coverage_rate,
        end_reason=end_reason,
    )


# ── drawing primitives ────────────────────────────────────────────────────────
def map_to_facecolors(grid_map: np.ndarray) -> np.ndarray:
    fc = np.zeros((*grid_map.shape, 4), dtype=float)
    fc[grid_map == -1] = mcolors.to_rgba(OBSTACLE_COLOR, 0.9)
    fc[grid_map ==  0] = mcolors.to_rgba(UNKNOWN_COLOR,  0.72)
    fc[grid_map ==  1] = mcolors.to_rgba(COVERED_COLOR,  0.78)
    return fc


def make_ground_mesh(env: MultiAgentHighLevelEnv):
    low_env = env.low_env
    edges = np.linspace(0.0, low_env.cfg.env_size, low_env.cfg.grid_size + 1)
    x, y = np.meshgrid(edges, edges, indexing="ij")
    return x, y, np.full_like(x, GROUND_Z)


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def draw_sphere(ax, center: Iterable[float], radius: float, color: str, zorder: int) -> None:
    cx, cy, cz = center
    u = np.linspace(0.0, 2.0 * np.pi, 12)
    v = np.linspace(0.0, np.pi, 8)
    x = cx + radius * np.outer(np.cos(u), np.sin(v))
    y = cy + radius * np.outer(np.sin(u), np.sin(v))
    z = cz + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, linewidth=0, shade=True, alpha=0.98, zorder=zorder)


def draw_circle3d(ax, center_xy, z, radius, color, alpha, linewidth, zorder) -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 72)
    x = center_xy[0] + radius * np.cos(t)
    y = center_xy[1] + radius * np.sin(t)
    ax.plot(x, y, np.full_like(t, z), color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)


def draw_uav(ax, pos_xy: np.ndarray, theta: float, altitude: float, scale: float,
             body_color: str, arm_color: str = UAV_ARM_COLOR) -> None:
    center = np.array([pos_xy[0], pos_xy[1], altitude], dtype=float)
    arm_len      = 1.75 * scale
    rotor_radius = 0.42 * scale
    body_radius  = 0.36 * scale

    draw_sphere(ax, center, body_radius, body_color, zorder=UAV_ZORDER)

    local_pts = np.array([[arm_len, 0.0], [-arm_len, 0.0],
                           [0.0, arm_len], [0.0, -arm_len]], dtype=float)
    world_pts = local_pts @ rot2(theta).T + pos_xy
    for a, b in [(0, 1), (2, 3)]:
        ax.plot(
            [world_pts[a, 0], world_pts[b, 0]],
            [world_pts[a, 1], world_pts[b, 1]],
            [altitude, altitude],
            color=arm_color, linewidth=2.8, zorder=UAV_ZORDER,
        )
    for rxy in world_pts:
        ax.plot([rxy[0], rxy[0]], [rxy[1], rxy[1]],
                [altitude - 0.15 * scale, altitude + 0.08 * scale],
                color="#374151", linewidth=1.4, zorder=UAV_ZORDER)
        draw_circle3d(ax, rxy, altitude + 0.08 * scale, rotor_radius,
                      "#0f172a", 0.9, 1.5, UAV_ZORDER)

    nose = pos_xy + rot2(theta) @ np.array([2.35 * scale, 0.0])
    ax.plot([center[0], nose[0]], [center[1], nose[1]],
            [altitude, altitude + 0.15 * scale],
            color=GUIDANCE_COLOR, linewidth=3.0, zorder=UAV_ZORDER)


def draw_agent_guidance(ax, guidance_world: np.ndarray | None,
                        guidance_color: str, altitude: float) -> None:
    if guidance_world is None:
        return
    wx, wy = float(guidance_world[0]), float(guidance_world[1])
    ax.scatter([wx], [wy], [0.25], c=guidance_color, marker="^",
               s=92, depthshade=False, zorder=GUIDANCE_ZORDER)
    ax.plot([wx, wx], [wy, wy], [0.25, altitude],
            color=guidance_color, alpha=0.45, linestyle="--",
            linewidth=1.4, zorder=GUIDANCE_ZORDER)


# ── scene composition ─────────────────────────────────────────────────────────
def draw_scene(
    ax,
    env: MultiAgentHighLevelEnv,
    frame: dict,
    trajectories: list[np.ndarray],   # one (T, 2) array per agent
    episode_idx: int,
    frame_idx: int,
    total_frames: int,
    result: RolloutResult,
    altitude: float,
    view_elev: float,
    view_azim: float,
) -> None:
    low_env = env.low_env
    ax.clear()
    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False

    # ground / coverage map
    x_mesh, y_mesh, z_mesh = make_ground_mesh(env)
    ax.plot_surface(
        x_mesh, y_mesh, z_mesh,
        facecolors=map_to_facecolors(frame["map"]),
        linewidth=0.2, edgecolor="#cbd5e1",
        shade=False, antialiased=False, zorder=GROUND_ZORDER,
    )

    agent_guidance: dict = frame.get("agent_guidance", {})
    n = len(frame["pos"])

    for i in range(n):
        body_color, traj_color, guide_color = _agent_colors(i)
        pos_xy = np.asarray(frame["pos"][i], dtype=float)
        theta  = float(frame["theta"][i])

        # detection circle
        draw_circle3d(ax, pos_xy, 0.12, low_env.cfg.det_radius,
                      body_color, 0.55, 1.8, SCAN_ZORDER)

        # UAV model
        draw_uav(ax, pos_xy, theta, altitude=altitude, scale=1.35,
                 body_color=body_color)

        # guidance point
        gw = agent_guidance.get(i)
        draw_agent_guidance(ax, gw, guide_color, altitude)

        # trajectory
        traj = trajectories[i]
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1],
                    np.full(len(traj), altitude * 0.78),
                    color=traj_color, linewidth=2.2, alpha=0.88,
                    zorder=TRAJECTORY_ZORDER)

    # start marker (shared spawn centre)
    ax.scatter([low_env.start_pos[0]], [low_env.start_pos[1]], [0.2],
               c=START_COLOR, marker="o", s=52, zorder=GUIDANCE_ZORDER)

    ax.set_xlim(0, low_env.cfg.env_size)
    ax.set_ylim(0, low_env.cfg.env_size)
    ax.set_zlim(0, max(altitude * 2.2, 8.0))
    ax.set_box_aspect((1, 1, 0.32))
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title(
        f"3D Multi-UAV Coverage | Episode {episode_idx} | "
        f"Frame {frame_idx + 1}/{total_frames} | "
        f"Coverage {result.coverage_rate * 100:.2f}%"
    )
    ax.grid(True, alpha=0.22)


# ── rendering ─────────────────────────────────────────────────────────────────
def render_rollout(
    env: MultiAgentHighLevelEnv,
    result: RolloutResult,
    episode_idx: int,
    output_path: Path,
    fps: int,
    dpi: int,
    frame_stride: int,
    altitude: float,
    view_elev: float,
    view_azim: float,
    show: bool,
) -> None:
    if not result.frames:
        raise RuntimeError("No rollout frames were recorded.")

    frames = result.frames[:: max(frame_stride, 1)]
    if frames[-1] is not result.frames[-1]:
        frames.append(result.frames[-1])

    n_agents = len(frames[0]["pos"])
    # build per-agent position arrays (one row per frame)
    all_positions = [
        np.array([f["pos"][i] for f in frames], dtype=float)
        for i in range(n_agents)
    ]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx: int):
        trajectories = [pos[: frame_idx + 1] for pos in all_positions]
        draw_scene(
            ax=ax,
            env=env,
            frame=frames[frame_idx],
            trajectories=trajectories,
            episode_idx=episode_idx,
            frame_idx=frame_idx,
            total_frames=len(frames),
            result=result,
            altitude=altitude,
            view_elev=view_elev,
            view_azim=view_azim,
        )
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer="pillow", fps=fps, dpi=dpi)
    print(f"3D GIF saved: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a 3D simulation for the multi-agent hierarchical coverage policy."
    )
    p.add_argument("--low-model",  default="Cover_multi/low_model/ppo_low_last.zip")
    p.add_argument("--high-model", default="Cover_multi/high_model/ppo_high_last.zip")
    p.add_argument("--output-dir", default="Cover_multi/simulation/renders")
    p.add_argument("--num-agents",     type=int,   default=3)
    p.add_argument("--episodes",       type=int,   default=5)
    p.add_argument("--seed",           type=int,   default=3000)
    p.add_argument("--grid-bins",      type=int,   default=5)
    p.add_argument("--option-horizon", type=int,   default=15)
    p.add_argument("--max-high-steps", type=int,   default=80)
    p.add_argument("--altitude",       type=float, default=5.5)
    p.add_argument("--fps",            type=int,   default=8)
    p.add_argument("--dpi",            type=int,   default=110)
    p.add_argument("--frame-stride",   type=int,   default=1)
    p.add_argument("--view-elev",      type=float, default=52.0)
    p.add_argument("--view-azim",      type=float, default=-55.0)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    low_model  = resolve_path(args.low_model)
    high_model = resolve_path(args.high_model)
    output_dir = resolve_path(args.output_dir)

    if not low_model.exists():
        raise FileNotFoundError(f"Low-level model not found: {low_model}")
    if not high_model.exists():
        raise FileNotFoundError(f"High-level model not found: {high_model}")

    env = build_env(
        low_model=low_model,
        num_agents=args.num_agents,
        grid_bins=args.grid_bins,
        option_horizon=args.option_horizon,
        max_high_steps=args.max_high_steps,
    )
    print(f"Loading high-level model: {high_model}")
    model = PPO.load(str(high_model), env=env, device="cpu")

    for ep in range(1, args.episodes + 1):
        seed = args.seed + ep
        print(f"Running episode {ep} (seed={seed}) ...")
        result = run_policy_rollout(env, model, seed=seed)
        print(
            f"  end={result.end_reason}, "
            f"coverage={result.coverage_rate * 100:.2f}%, "
            f"high_steps={result.high_steps}, "
            f"frames={len(result.frames)}, "
            f"reward={result.total_reward:.3f}"
        )
        output_path = output_dir / f"multi_hl_3d_episode_{ep}.gif"
        render_rollout(
            env=env,
            result=result,
            episode_idx=ep,
            output_path=output_path,
            fps=args.fps,
            dpi=args.dpi,
            frame_stride=args.frame_stride,
            altitude=args.altitude,
            view_elev=args.view_elev,
            view_azim=args.view_azim,
            show=args.show,
        )

    env.close()


if __name__ == "__main__":
    main()
