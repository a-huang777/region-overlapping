from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import gymnasium as gym

# 导入自定义环境
from env import SimpleCartPoleEnv

if __name__ == "__main__":
    # 创建日志和模型目录
    log_dir = "./logs"
    models_dir = "./models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 创建自定义环境的并行环境
    env = make_vec_env(
        "SimpleCartPole-v0", 
        n_envs=20, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": None}  # 明确设置render_mode避免警告
    )
    
    # 创建PPO模型，使用CUDA并启用TensorBoard日志
    model = PPO(
        "MlpPolicy", 
        env, 
        device="cuda", 
        verbose=1,
        tensorboard_log=log_dir  # 指定TensorBoard日志目录
    )
    
    # 训练模型，指定日志名称
    model.learn(
        total_timesteps=10000000,
        tb_log_name="PPO_PID_Controller"  # TensorBoard日志名称
    )
    
    # 保存模型
    model.save(os.path.join(models_dir, "ppo_pid_final"))
    
    print(f"训练完成！")
    print(f"模型已保存到: {os.path.join(models_dir, 'ppo_pid_final')}")
    print(f"TensorBoard日志位于: {log_dir}")
    
    # 关闭环境
    env.close()
