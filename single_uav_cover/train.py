import os

# Avoid MKL + libiomp5 symbol clash: "undefined symbol: __kmpc_global_thread_num"
# Must run before NumPy / PyTorch / SB3 load MKL.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import Config, MultiUAVCoverageEnv


def get_next_version_dir(checkpoint_base_dir: str):
    os.makedirs(checkpoint_base_dir, exist_ok=True)
    max_version = -1
    for name in os.listdir(checkpoint_base_dir):
        if not name.startswith("version_"):
            continue
        suffix = name[len("version_"):]
        if suffix.isdigit():
            max_version = max(max_version, int(suffix))

    version_name = f"version_{max_version + 1}"
    version_dir = os.path.join(checkpoint_base_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    return version_dir, version_name


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


def train():
    n_envs = 8
    total_timesteps = 10_000_000
    save_freq_steps = 200_000

    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_base_dir = os.path.join(base_dir, "check_point")
    version_dir, version_name = get_next_version_dir(checkpoint_base_dir)
    model_dir = os.path.join(version_dir, "model")
    tensorboard_dir = os.path.join(version_dir, "tensorboard")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    print(f"Checkpoint version: {version_name}")
    print(f"Model dir: {model_dir}")
    print(f"TensorBoard dir: {tensorboard_dir}")

    def make_env(rank: int):
        def _init():
            env_cfg = Config()
            env_cfg.num_agents = 1
            env = SingleUAVSB3Wrapper(MultiUAVCoverageEnv(env_cfg))
            env = Monitor(env)
            env.reset(seed=42 + rank)
            return env

        return _init

    env_fns = [make_env(i) for i in range(n_envs)]
    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=tensorboard_dir,
        verbose=1,
        device="cpu",
    )

    # VecEnv 下 callback 的触发频率按 "env.step 调用次数" 计，需按并行环境数折算。
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq_steps // n_envs, 1),
        save_path=model_dir,
        name_prefix="ppo_single_uav_step",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="ppo_single_uav",
        progress_bar=True,
        callback=checkpoint_callback,
    )

    final_model_path = os.path.join(model_dir, "ppo_single_uav_final")
    model.save(final_model_path)
    print(f"Saved final model: {final_model_path}.zip")

    env.close()


if __name__ == "__main__":
    train()
