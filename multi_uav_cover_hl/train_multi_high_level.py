"""
多智能体集中式高层 PPO 训练脚本（阶段一）

用法（在 Cover/ 目录下运行）：
    python train_multi_high_level.py

或指定参数：
    python train_multi_high_level.py \
        --low_model low_model/ppo_model_save.zip \
        --num_agents 3 \
        --n_envs 4 \
        --total_timesteps 2000000
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_THIS_DIR, "simulation", ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# matplotlib 依赖的 libstdc++ 版本高于系统库，用 conda 环境的版本覆盖
_conda_lib = os.path.join(sys.prefix, "lib")
_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
if _conda_lib not in _ld_path:
    os.environ["LD_LIBRARY_PATH"] = f"{_conda_lib}:{_ld_path}"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# 把当前工程目录加入 sys.path，使本地 env.py / multi_high_level_env.py 可以被找到
_REPO_ROOT = _THIS_DIR
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from multi_high_level_env import MultiAgentHighLevelEnv, MultiHighLevelConfig
from env import Config


def get_next_version_dir(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    max_v = -1
    for name in os.listdir(base_dir):
        if name.startswith("version_") and name[8:].isdigit():
            max_v = max(max_v, int(name[8:]))
    version_name = f"version_{max_v + 1}"
    version_dir = os.path.join(base_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    return version_dir, version_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--low_model", default="low_model/ppo_model_save.zip")
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--total_timesteps", type=int, default=2_000_000)
    p.add_argument("--grid_bins", type=int, default=5)
    p.add_argument("--option_horizon", type=int, default=15)
    p.add_argument("--max_high_steps", type=int, default=80)
    p.add_argument("--use_subproc", action="store_true", default=False,
                   help="使用 SubprocVecEnv（多进程）；默认 DummyVecEnv（单进程，调试更方便）")
    return p.parse_args()


def train():
    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    low_model_path = args.low_model
    if not os.path.isabs(low_model_path):
        low_model_path = os.path.join(_REPO_ROOT, low_model_path)
    if not os.path.exists(low_model_path):
        raise FileNotFoundError(f"low-level model not found: {low_model_path}")

    hl_cfg = MultiHighLevelConfig(
        grid_bins=args.grid_bins,
        option_horizon=args.option_horizon,
        max_high_steps=args.max_high_steps,
    )

    checkpoint_base = os.path.join(base_dir, "check_point_multi_high_level")
    version_dir, version_name = get_next_version_dir(checkpoint_base)
    model_dir = os.path.join(version_dir, "model")
    tb_dir = os.path.join(version_dir, "tensorboard")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    print(f"version       : {version_name}")
    print(f"num_agents    : {args.num_agents}")
    print(f"grid_bins     : {hl_cfg.grid_bins}x{hl_cfg.grid_bins} = {hl_cfg.grid_bins**2} regions")
    print(f"option_horizon: {hl_cfg.option_horizon}")
    print(f"max_high_steps: {hl_cfg.max_high_steps}")
    print(f"n_envs        : {args.n_envs}")
    print(f"total_steps   : {args.total_timesteps:,}")
    print(f"low model     : {low_model_path}")
    print(f"model_dir     : {model_dir}")
    print(f"tensorboard   : {tb_dir}")

    def make_env(rank: int):
        def _init():
            env_cfg = Config()
            return MultiAgentHighLevelEnv(
                low_level_model_path=low_model_path,
                env_cfg=env_cfg,
                hl_cfg=hl_cfg,
                num_agents=args.num_agents,
                deterministic_low_level=True,
            )
        return _init

    env_fns = [make_env(i) for i in range(args.n_envs)]
    if args.use_subproc and args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        device="cpu",
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tb_dir,
        verbose=1,
    )

    save_freq = max(args.total_timesteps // 20, 50_000)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq // args.n_envs,
        save_path=model_dir,
        name_prefix="ppo_multi_hl",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    final_path = os.path.join(model_dir, "ppo_multi_hl_final.zip")
    model.save(final_path)
    print(f"saved: {final_path}")

    vec_env.close()


if __name__ == "__main__":
    train()
