import os
from dataclasses import dataclass

# Avoid MKL/OpenMP symbol clash in some conda envs.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from high_level_env import HighLevelConfig, HighLevelGuidanceEnv


@dataclass
class TrainHighLevelConfig:
    low_level_model_path: str = "low_model/ppo_model_save.zip"
    total_timesteps: int = 2_000_000
    n_envs: int = 4
    save_freq_steps: int = 100_000
    use_subproc: bool = True
    grid_bins: int = 5          #action space
    option_horizon: int = 10    #steps
    max_high_steps: int = 100    #上层每回合宏观步上限（truncated）


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


def train_high_level():
    cfg = TrainHighLevelConfig()
    hl_cfg = HighLevelConfig()
    hl_cfg.grid_bins = cfg.grid_bins
    hl_cfg.option_horizon = cfg.option_horizon
    hl_cfg.max_high_steps = cfg.max_high_steps

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(cfg.low_level_model_path):
        low_model_path = os.path.join(base_dir, cfg.low_level_model_path)
    else:
        low_model_path = cfg.low_level_model_path
    if not os.path.exists(low_model_path):
        raise FileNotFoundError(f"low-level model not found: {low_model_path}")

    checkpoint_base_dir = os.path.join(base_dir, "check_point_high_level")
    version_dir, version_name = get_next_version_dir(checkpoint_base_dir)
    model_dir = os.path.join(version_dir, "model")
    tensorboard_dir = os.path.join(version_dir, "tensorboard")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    print(f"High-level checkpoint version: {version_name}")
    print(f"Low-level model path: {low_model_path}")
    print(f"Grid bins: {hl_cfg.grid_bins}x{hl_cfg.grid_bins}")
    print(f"Option horizon: {hl_cfg.option_horizon}")
    print(f"Max high-level steps per episode: {hl_cfg.max_high_steps}")
    print(f"Model dir: {model_dir}")
    print(f"TensorBoard dir: {tensorboard_dir}")

    def make_env(_rank: int):
        def _init():
            return HighLevelGuidanceEnv(
                low_level_model_path=low_model_path,
                hl_cfg=hl_cfg,
                deterministic_low_level=True,
            )

        return _init

    env_fns = [make_env(i) for i in range(cfg.n_envs)]
    if cfg.n_envs > 1 and cfg.use_subproc:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    env = VecMonitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=tensorboard_dir,
        verbose=1,
        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg.save_freq_steps // max(cfg.n_envs, 1), 1),
        save_path=model_dir,
        name_prefix="ppo_high_level_step",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        tb_log_name="ppo_high_level",
        progress_bar=True,
        callback=checkpoint_callback,
    )

    final_model_path = os.path.join(model_dir, "ppo_high_level_final")
    model.save(final_model_path)
    print(f"Saved final high-level model: {final_model_path}.zip")

    env.close()


if __name__ == "__main__":
    train_high_level()
