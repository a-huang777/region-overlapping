import os

# Avoid MKL + libiomp5 symbol clash: "undefined symbol: __kmpc_global_thread_num"
# Must run before NumPy / PyTorch / SB3 load MKL.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Wedge
from stable_baselines3 import PPO

from env import Config, MultiUAVCoverageEnv


class SingleUAVSB3Wrapper(gym.Wrapper):
    """
    将原始环境输出从 (1, obs_dim)/(1, 2) 适配到 SB3 单智能体格式:
      obs: (obs_dim,)
      action: (2,)
      reward: float
    """

    def __init__(self, env: MultiUAVCoverageEnv):
        super().__init__(env)
        if env.n != 1:
            raise ValueError(f"SingleUAVSB3Wrapper requires env.n == 1, got {env.n}")

        obs_dim = int(env.observation_space.shape[-1])
        act_dim = int(env.action_space.shape[-1])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[0].astype(np.float32), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_scalar = float(reward[0])
        info["reward_components"] = info.get("agent_0", {})
        return obs[0].astype(np.float32), reward_scalar, terminated, truncated, info


def run_and_save_gif(base_env, wrapped_env, model, episode_idx, gif_dir="test_gifs"):
    print(f"\n--- 开始录制第 {episode_idx} 个回合 ---")
    os.makedirs(gif_dir, exist_ok=True)

    obs, reset_info = wrapped_env.reset()
    history = []
    terminated = False
    truncated = False
    step_count = 0
    total_reward = 0.0

    history.append(
        {
            "pos": base_env.agents_pos.copy(),
            "theta": base_env.agents_theta.copy(),
            "map": base_env.global_map.copy(),
            "guidance_points": reset_info.get("agent_0", {}).get("guidance_points", []),
        }
    )

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)

        total_reward += float(reward)
        step_count += 1

        history.append(
            {
                "pos": base_env.agents_pos.copy(),
                "theta": base_env.agents_theta.copy(),
                "map": base_env.global_map.copy(),
                "guidance_points": info.get("agent_0", {}).get("guidance_points", []),
            }
        )

    print(f"回合结束。总步数: {step_count}, 累计奖励: {total_reward:.2f}")
    print("正在渲染并生成 GIF，请稍候...")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, base_env.cfg.env_size)
    ax.set_ylim(0, base_env.cfg.env_size)
    ax.set_aspect("equal")
    ax.set_title(f"SB3 Single-UAV Coverage - Episode {episode_idx}", fontsize=14)

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
    guidance_scatter = ax.scatter([], [], c="red", marker="^", s=100, label="Guidance Points")
    ax.legend(loc="upper right")

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

    gif_path = os.path.join(gif_dir, f"sb3_coverage_test_{episode_idx}.gif")
    ani.save(gif_path, writer="pillow", fps=8)
    print(f"✅ GIF 已保存至: {gif_path}")
    plt.close(fig)


if __name__ == "__main__":
    # 1) 创建环境（单无人机）
    env_cfg = Config()
    env_cfg.num_agents = 1
    env_cfg.infer_mode = True
    env_cfg.max_steps = 500
    base_env = MultiUAVCoverageEnv(env_cfg)
    wrapped_env = SingleUAVSB3Wrapper(base_env)

    # 2) 加载 SB3 训练模型（请按你的实际路径修改）
    model_path = "check_point/version_0/model/ppo_model_save.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    print(f"成功加载 SB3 模型: {model_path}")
    model = PPO.load(model_path, env=wrapped_env, device="cpu")

    # 3) 运行并生成 GIF
    for i in range(1, 6):
        run_and_save_gif(base_env, wrapped_env, model, episode_idx=i, gif_dir="test_gifs")

    wrapped_env.close()
    print("\n🎉 全部 5 张 SB3 演示 GIF 生成完成！")
