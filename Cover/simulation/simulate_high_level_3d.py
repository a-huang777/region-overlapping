"""
Run the trained high-level policy and render a simple 3D simulation.

The learned environment is still 2D. This visualizer lifts the UAV to a fixed
altitude and renders the coverage map, guidance point, trajectory, and a small
quadrotor-style UAV model in a 3D scene.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(SIM_DIR / ".mplconfig"))
(SIM_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from stable_baselines3 import PPO

from env import Config
from high_level_env import HighLevelConfig, HighLevelGuidanceEnv


COVERED_COLOR = "#bfdbfe"
UNKNOWN_COLOR = "#f8fafc"
OBSTACLE_COLOR = "#4b5563"
DETECTION_COLOR = "#1e40af"
GUIDANCE_COLOR = "#dc2626"
UAV_BODY_COLOR = "#2563eb"
UAV_ARM_COLOR = "#111827"
TRAJECTORY_COLOR = "#f97316"
START_COLOR = "#1d4ed8"
GROUND_Z = -0.04
GROUND_ZORDER = 0
SCAN_ZORDER = 18
TRAJECTORY_ZORDER = 24
GUIDANCE_ZORDER = 30
UAV_ZORDER = 40


@dataclass
class RolloutResult:
    frames: list[dict]
    high_steps: int
    total_reward: float
    coverage_rate: float
    end_reason: str


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_env(low_model: Path, grid_bins: int, option_horizon: int, max_high_steps: int) -> HighLevelGuidanceEnv:
    env_cfg = Config()
    env_cfg.coverage_goal = 0.98
    env_cfg.infer_mode = False

    hl_cfg = HighLevelConfig()
    hl_cfg.grid_bins = grid_bins
    hl_cfg.option_horizon = option_horizon
    hl_cfg.max_high_steps = max_high_steps

    return HighLevelGuidanceEnv(
        low_level_model_path=str(low_model),
        env_cfg=env_cfg,
        hl_cfg=hl_cfg,
        deterministic_low_level=True,
        infer_mode=False,
    )


def run_policy_rollout(
    env: HighLevelGuidanceEnv,
    high_model: PPO,
    seed: int,
) -> RolloutResult:
    env.record_low_frames = True
    env.low_frame_buffer.clear()

    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    high_steps = 0
    total_reward = 0.0
    latest_info: dict = {}
    safety_cap = max(int(env.hl_cfg.max_high_steps), 1) + 32

    while not (terminated or truncated) and high_steps < safety_cap:
        action, _ = high_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, latest_info = env.step(int(action))
        total_reward += float(reward)
        high_steps += 1

    env.record_low_frames = False

    coverage_rate = float(latest_info.get("coverage_rate", env._coverage_stats()[0]))
    if latest_info.get("coverage_success"):
        end_reason = "coverage_goal_met"
    elif truncated:
        end_reason = "high_level_truncated"
    elif terminated:
        end_reason = str(latest_info.get("option_end_reason", "terminated"))
    else:
        end_reason = "safety_cap"

    return RolloutResult(
        frames=list(env.low_frame_buffer),
        high_steps=high_steps,
        total_reward=total_reward,
        coverage_rate=coverage_rate,
        end_reason=end_reason,
    )


def map_to_facecolors(grid_map: np.ndarray) -> np.ndarray:
    facecolors = np.zeros((*grid_map.shape, 4), dtype=float)
    facecolors[grid_map == -1] = mcolors.to_rgba(OBSTACLE_COLOR, 0.9)
    facecolors[grid_map == 0] = mcolors.to_rgba(UNKNOWN_COLOR, 0.72)
    facecolors[grid_map == 1] = mcolors.to_rgba(COVERED_COLOR, 0.78)
    return facecolors


def make_ground_mesh(env: HighLevelGuidanceEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_env = env.low_env
    edges = np.linspace(0.0, low_env.cfg.env_size, low_env.cfg.grid_size + 1)
    x, y = np.meshgrid(edges, edges, indexing="ij")
    z = np.full_like(x, GROUND_Z)
    return x, y, z


def rot2(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def draw_sphere(ax, center: Iterable[float], radius: float, color: str, zorder: int) -> None:
    cx, cy, cz = center
    u = np.linspace(0.0, 2.0 * np.pi, 12)
    v = np.linspace(0.0, np.pi, 8)
    x = cx + radius * np.outer(np.cos(u), np.sin(v))
    y = cy + radius * np.outer(np.sin(u), np.sin(v))
    z = cz + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, linewidth=0, shade=True, alpha=0.98, zorder=zorder)


def draw_circle3d(
    ax,
    center_xy: np.ndarray,
    z: float,
    radius: float,
    color: str,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 72)
    x = center_xy[0] + radius * np.cos(t)
    y = center_xy[1] + radius * np.sin(t)
    ax.plot(x, y, np.full_like(t, z), color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)


def draw_uav(ax, pos_xy: np.ndarray, theta: float, altitude: float, scale: float) -> None:
    center = np.array([pos_xy[0], pos_xy[1], altitude], dtype=float)
    arm_len = 1.75 * scale
    rotor_radius = 0.42 * scale
    body_radius = 0.36 * scale

    draw_sphere(ax, center, body_radius, UAV_BODY_COLOR, zorder=UAV_ZORDER)

    local_points = np.array(
        [
            [arm_len, 0.0],
            [-arm_len, 0.0],
            [0.0, arm_len],
            [0.0, -arm_len],
        ],
        dtype=float,
    )
    world_points = local_points @ rot2(theta).T + pos_xy
    pairs = [(0, 1), (2, 3)]
    for a, b in pairs:
        ax.plot(
            [world_points[a, 0], world_points[b, 0]],
            [world_points[a, 1], world_points[b, 1]],
            [altitude, altitude],
            color=UAV_ARM_COLOR,
            linewidth=2.8,
            zorder=UAV_ZORDER,
        )

    for rotor_xy in world_points:
        ax.plot(
            [rotor_xy[0], rotor_xy[0]],
            [rotor_xy[1], rotor_xy[1]],
            [altitude - 0.15 * scale, altitude + 0.08 * scale],
            color="#374151",
            linewidth=1.4,
            zorder=UAV_ZORDER,
        )
        draw_circle3d(ax, rotor_xy, altitude + 0.08 * scale, rotor_radius, "#0f172a", 0.9, 1.5, UAV_ZORDER)

    nose = pos_xy + rot2(theta) @ np.array([2.35 * scale, 0.0])
    ax.plot(
        [center[0], nose[0]],
        [center[1], nose[1]],
        [altitude, altitude + 0.15 * scale],
        color=GUIDANCE_COLOR,
        linewidth=3.0,
        zorder=UAV_ZORDER,
    )


def draw_guidance(ax, guidance_points: list[dict], altitude: float) -> None:
    if not guidance_points:
        return
    for item in guidance_points:
        wx, wy = item["world"]
        ax.scatter([wx], [wy], [0.25], c=GUIDANCE_COLOR, marker="^", s=92, depthshade=False, zorder=GUIDANCE_ZORDER)
        ax.plot(
            [wx, wx],
            [wy, wy],
            [0.25, altitude],
            color=GUIDANCE_COLOR,
            alpha=0.45,
            linestyle="--",
            linewidth=1.4,
            zorder=GUIDANCE_ZORDER,
        )


def draw_scene(
    ax,
    env: HighLevelGuidanceEnv,
    frame: dict,
    trajectory: np.ndarray,
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

    x_mesh, y_mesh, z_mesh = make_ground_mesh(env)
    ax.plot_surface(
        x_mesh,
        y_mesh,
        z_mesh,
        facecolors=map_to_facecolors(frame["map"]),
        linewidth=0.2,
        edgecolor="#cbd5e1",
        shade=False,
        antialiased=False,
        zorder=GROUND_ZORDER,
    )

    pos = np.asarray(frame["pos"][0], dtype=float)
    theta = float(frame["theta"][0])
    draw_circle3d(ax, pos, 0.12, low_env.cfg.det_radius, DETECTION_COLOR, 0.75, 2.0, SCAN_ZORDER)
    draw_uav(ax, pos, theta, altitude=altitude, scale=1.35)
    draw_guidance(ax, frame.get("guidance_points", []), altitude=altitude)

    if len(trajectory) > 1:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            np.full(len(trajectory), altitude * 0.78),
            color=TRAJECTORY_COLOR,
            linewidth=2.4,
            alpha=0.9,
            zorder=TRAJECTORY_ZORDER,
        )

    ax.scatter([low_env.start_pos[0]], [low_env.start_pos[1]], [0.2], c=START_COLOR, marker="o", s=52, zorder=GUIDANCE_ZORDER)

    ax.set_xlim(0, low_env.cfg.env_size)
    ax.set_ylim(0, low_env.cfg.env_size)
    ax.set_zlim(0, max(altitude * 2.2, 8.0))
    ax.set_box_aspect((1, 1, 0.32))
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title(
        f"3D Coverage Simulation | Episode {episode_idx} | "
        f"Frame {frame_idx + 1}/{total_frames} | Coverage {result.coverage_rate * 100:.2f}%"
    )
    ax.grid(True, alpha=0.22)


def render_rollout(
    env: HighLevelGuidanceEnv,
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

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    positions = np.array([item["pos"][0] for item in frames], dtype=float)

    def update(frame_idx: int):
        trajectory = positions[: frame_idx + 1]
        draw_scene(
            ax=ax,
            env=env,
            frame=frames[frame_idx],
            trajectory=trajectory,
            episode_idx=episode_idx,
            frame_idx=frame_idx,
            total_frames=len(frames),
            result=result,
            altitude=altitude,
            view_elev=view_elev,
            view_azim=view_azim,
        )
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer="pillow", fps=fps, dpi=dpi)
    print(f"3D GIF saved: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 3D simulation for the hierarchical coverage policy.")
    parser.add_argument("--low-model", default="low_model/ppo_model_save.zip", help="Path to the trained low-level PPO model.")
    parser.add_argument(
        "--high-model",
        default="check_point_high_level/version_3/model/ppo_high_level_final.zip",
        help="Path to the trained high-level PPO model.",
    )
    parser.add_argument("--output-dir", default="simulation/renders", help="Directory for generated GIF files.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to render.")
    parser.add_argument("--seed", type=int, default=3000, help="Base random seed.")
    parser.add_argument("--grid-bins", type=int, default=5, help="High-level grid bins per axis.")
    parser.add_argument("--option-horizon", type=int, default=10, help="Low-level steps per high-level decision.")
    parser.add_argument("--max-high-steps", type=int, default=100, help="Maximum high-level decisions per episode.")
    parser.add_argument("--altitude", type=float, default=5.5, help="Fixed rendered UAV altitude in meters.")
    parser.add_argument("--fps", type=int, default=8, help="Output GIF frame rate.")
    parser.add_argument("--dpi", type=int, default=110, help="Output GIF resolution.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Render every Nth recorded low-level frame.")
    parser.add_argument("--view-elev", type=float, default=52.0, help="3D camera elevation.")
    parser.add_argument("--view-azim", type=float, default=-55.0, help="3D camera azimuth.")
    parser.add_argument("--show", action="store_true", help="Show an interactive Matplotlib window after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    low_model = resolve_path(args.low_model)
    high_model = resolve_path(args.high_model)
    output_dir = resolve_path(args.output_dir)

    if not low_model.exists():
        raise FileNotFoundError(f"Low-level model not found: {low_model}")
    if not high_model.exists():
        raise FileNotFoundError(f"High-level model not found: {high_model}")

    env = build_env(
        low_model=low_model,
        grid_bins=args.grid_bins,
        option_horizon=args.option_horizon,
        max_high_steps=args.max_high_steps,
    )
    print(f"Loading high-level model: {high_model}")
    model = PPO.load(str(high_model), env=env, device="cpu")

    for episode_idx in range(1, args.episodes + 1):
        seed = args.seed + episode_idx
        print(f"Running episode {episode_idx} with seed {seed}...")
        result = run_policy_rollout(env, model, seed=seed)
        print(
            f"Episode {episode_idx}: end={result.end_reason}, "
            f"coverage={result.coverage_rate * 100:.2f}%, "
            f"high_steps={result.high_steps}, frames={len(result.frames)}, "
            f"reward={result.total_reward:.3f}"
        )
        output_path = output_dir / f"high_level_3d_episode_{episode_idx}.gif"
        render_rollout(
            env=env,
            result=result,
            episode_idx=episode_idx,
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
