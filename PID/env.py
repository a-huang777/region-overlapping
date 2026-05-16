import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SimpleCartPoleEnv(gym.Env):
    """
    RL实现PID控制器
    状态空间：[当前速度_vx, 当前速度_vy, 目标速度_vx, 目标速度_vy]
    动作空间：[加速度_x, 加速度_y] - 连续动作
    目标：让当前速度快速跟随目标速度
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super(SimpleCartPoleEnv, self).__init__()
        
        self.episode = 0
        self.current_step = 0
        self.max_steps = 500
        
        # 动作空间和状态空间维度
        self.action_dim = 2    # 动作维度（加速度x, 加速度y）
        self.obs_dim = 4       # 状态维度（当前vx, 当前vy, 目标vx, 目标vy）
        
        # 目标速度（会随机生成）
        self.target_speed = np.array([0.0, 0.0], dtype=np.float32)
        # 当前速度
        self.current_speed = np.array([0.0, 0.0], dtype=np.float32)
        
        # 动作空间：2个维度的连续动作 [accel_x, accel_y]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        # 状态空间：4个维度 [current_vx, current_vy, target_vx, target_vy]
        # 观察空间范围根据实际速度范围设置
        self.observation_space = spaces.Box(
            low=np.array([-100.0, -100.0, -100.0, -100.0], dtype=np.float32),
            high=np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 速度范围（用于归一化）
        self.max_speed = 100.0
        self.max_accel = 1.0  # 最大加速度
        
        # 渲染模式
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
    
    def reset(self, seed=None, options=None):
        """
        重置环境
        """
        super().reset(seed=seed)
        
        # 随机生成新的目标速度
        self.target_speed = self.np_random.uniform(low=-50.0, high=50.0, size=(2,)).astype(np.float32)
        
        # 重置当前速度
        self.current_speed = self.np_random.uniform(low=-5.0, high=5.0, size=(2,)).astype(np.float32)
        
        self.episode += 1
        self.current_step = 0
        self.prev_action = np.zeros(2, dtype=np.float32)  # 初始化上一步的动作
        
        # 构建状态：[当前速度, 目标速度]
        observation = np.concatenate([self.current_speed, self.target_speed]).astype(np.float32)
        
        info = {}
        return observation, info
    
    def step(self, action):
        """
        执行动作
        """
        reward = 0
        done = False
        self.current_step += 1
        
        before_speed_error = np.linalg.norm(self.target_speed - self.current_speed)
        # 动作是加速度，应用到当前速度上
        for _ in range(10):
            self.current_speed += action * 0.1  # 时间步长
        after_speed_error = np.linalg.norm(self.target_speed - self.current_speed)
        # 限制当前速度在合理范围内
        self.current_speed = np.clip(self.current_speed, -self.max_speed, self.max_speed)
        
        # 构建新状态：[当前速度, 目标速度]
        next_state = np.concatenate([self.current_speed, self.target_speed]).astype(np.float32)
        
        # 计算奖励
        reward = 0.1*(before_speed_error - after_speed_error)
        
        # 2. 动作平滑性奖励 - 避免动作变化过大
        if self.current_step > 1:
            action_diff = np.linalg.norm(action - self.prev_action)
            reward -= 0.01 * action_diff
        
        # 3. 达到目标奖励 - 如果速度已经很接近目标速度
        speed_error = np.linalg.norm(self.target_speed - self.current_speed)
        if speed_error < 1.0:
            reward += 1  # 达到目标的额外奖励
        
        # 4. 步数惩罚 - 鼓励快速达到目标
        reward -= 0.1  # 每步小惩罚，鼓励快速收敛
        
        for speed in self.current_speed:
            if abs(speed) > self.max_speed:
                reward -= 10.0  # 速度超出范围的惩罚
                done = True
        # 检查是否终止
        terminated = done
        truncated = self.current_step >= self.max_steps
        
        # 保存当前动作用于下一步的平滑性计算
        self.prev_action = action.copy()
        
        return next_state, reward, terminated, truncated, {}
    
    def render(self):
        """
        渲染环境
        """
        if self.render_mode is None:
            return
        
        if self.render_mode == "ansi":
            # 文本渲染
            s = f"\nEpisode: {self.episode}, Step: {self.current_step}"
            s += f"\n当前速度: [{self.current_speed[0]:.2f}, {self.current_speed[1]:.2f}]"
            s += f"\n目标速度: [{self.target_speed[0]:.2f}, {self.target_speed[1]:.2f}]"
            speed_error = np.linalg.norm(self.target_speed - self.current_speed)
            s += f"\n速度误差: {speed_error:.2f}"
            return s
    
    def close(self):
        """
        关闭环境
        """
        pass

# 注册自定义环境
try:
    gym.register(
        id="SimpleCartPole-v0",
        entry_point="env:SimpleCartPoleEnv",
        max_episode_steps=500,
    )
except:
    pass  # 环境已注册