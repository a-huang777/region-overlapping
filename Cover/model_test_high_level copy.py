"""
测试上层 PPO：rollout 并保存 GIF（与下层 model_test 风格一致，按低层步录制画面）。
请保证 grid_bins / option_horizon / 下层模型路径与训练上层时一致。
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Wedge
from stable_baselines3 import PPO

from env import Config
from high_level_env import HighLevelConfig, HighLevelGuidanceEnv


class TrainLikeConfig:
    low_level_model_path = "low_model/ppo_model_save.zip"
    grid_bins = 5
    option_horizon = 10
    high_level_model_path = "check_point_high_level/version_3/model/ppo_high_level_final.zip"


def run_and_save_gif(high_env: HighLevelGuidanceEnv, high_model, episode_idx: int, gif_dir="test_gifs_high"):
    base_env = high_env.low_env
    print(f"\n--- 开始录制上层测试 Episode {episode_idx} ---")
    os.makedirs(gif_dir, exist_ok=True)

    high_env.record_low_frames = True
    high_env.low_frame_buffer.clear()

    obs, _ = high_env.reset(seed=2000 + episode_idx)
    terminated = False
    truncated = False
    high_step = 0
    total_high_reward = 0.0
    # 与 env.hl_cfg.max_high_steps 对齐；略留冗余防止异常情况死循环
    env_macro_limit = max(int(high_env.hl_cfg.max_high_steps), 1)
    safety_cap = env_macro_limit + 32

    while not (terminated or truncated) and high_step < safety_cap:
        action, _ = high_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = high_env.step(int(action))
        total_high_reward += float(reward)
        high_step += 1

    history = list(high_env.low_frame_buffer)
    high_env.record_low_frames = False

    if len(history) == 0:
        print("无录制帧，跳过 GIF")
        return

    print(f"上层步数: {high_step}, 累计上层奖励: {total_high_reward:.3f}, 低层帧数: {len(history)}")
    print("正在渲染 GIF...")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, base_env.cfg.env_size)
    ax.set_ylim(0, base_env.cfg.env_size)
    ax.set_aspect("equal")
    ax.set_title(f"High-Level + Low-Level Rollout - Episode {episode_idx}", fontsize=14)

    cmap = ListedColormap(["#808080", "#FFFFFF", "#C8E6C9"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    img_map = ax.imshow(
        history[0]["map"].T,
        origin="lower",
        extent=[0, base_env.cfg.env_size, 0, base_env.cfg.env_size],
        cmap=cmap,
        norm=norm,
        alpha=0.6,
    )

    ax.add_patch(Circle(base_env.start_pos, base_env.cfg.safe_radius, color="blue", alpha=0.15, label="Start Zone"))
    ax.add_patch(Circle(base_env.target_pos, base_env.cfg.safe_radius, color="red", alpha=0.15, label="Target Zone"))
    ax.plot(base_env.start_pos[0], base_env.start_pos[1], "bo")
    ax.plot(base_env.target_pos[0], base_env.target_pos[1], "r*", markersize=12)

    body = Circle(history[0]["pos"][0], base_env.cfg.uav_radius, color="#1f77b4")
    det_circle = Circle(history[0]["pos"][0], base_env.cfg.det_radius, color="#1f77b4", alpha=0.2)
    heading = Wedge(
        history[0]["pos"][0],
        base_env.cfg.uav_radius * 2,
        np.degrees(history[0]["theta"][0]) - 15,
        np.degrees(history[0]["theta"][0]) + 15,
        color="black",
        alpha=0.8,
    )
    ax.add_patch(det_circle)
    ax.add_patch(body)
    ax.add_patch(heading)
    guidance_scatter = ax.scatter([], [], c="red", marker="^", s=100, label="Guidance (from high-level)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])

    def update(frame):
        state = history[frame]
        pos = state["pos"][0]
        theta = state["theta"][0]
        img_map.set_data(state["map"].T)
        body.center = pos
        det_circle.center = pos
        heading.set_center(pos)
        heading.set_theta1(np.degrees(theta) - 15)
        heading.set_theta2(np.degrees(theta) + 15)

        guidance_points = state.get("guidance_points", [])
        if guidance_points:
            gxy = np.array([item["grid"] for item in guidance_points], dtype=np.float32)
            gxy_world = (gxy + 0.5) * base_env.cfg.grid_res
            guidance_scatter.set_offsets(gxy_world)
        else:
            guidance_scatter.set_offsets(np.empty((0, 2)))

        return [img_map, body, det_circle, heading, guidance_scatter]

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=200, blit=False)
    gif_path = os.path.join(gif_dir, f"high_level_test_{episode_idx}.gif")
    ani.save(gif_path, writer="pillow", fps=8)
    print(f"GIF 已保存: {gif_path}")
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = TrainLikeConfig()

    low_path = (
        cfg.low_level_model_path
        if os.path.isabs(cfg.low_level_model_path)
        else os.path.join(base_dir, cfg.low_level_model_path)
    )
    high_path = (
        cfg.high_level_model_path
        if os.path.isabs(cfg.high_level_model_path)
        else os.path.join(base_dir, cfg.high_level_model_path)
    )

    if not os.path.exists(low_path):
        raise FileNotFoundError(f"下层模型不存在: {low_path}")
    if not os.path.exists(high_path):
        raise FileNotFoundError(f"上层模型不存在: {high_path}")

    hl_cfg = HighLevelConfig()
    hl_cfg.grid_bins = cfg.grid_bins
    hl_cfg.option_horizon = cfg.option_horizon

    env = HighLevelGuidanceEnv(
        low_level_model_path=low_path,
        env_cfg=Config(),
        hl_cfg=hl_cfg,
        deterministic_low_level=True,
        infer_mode=True,
    )

    print(f"加载上层模型: {high_path}")
    model = PPO.load(high_path, env=env, device="cpu")

    for i in range(1, 6):
        run_and_save_gif(env, model, episode_idx=i, gif_dir="test_gifs_high")

    print("\n全部 5 个上层测试 GIF 已生成（test_gifs_high/）")


if __name__ == "__main__":
    main()
