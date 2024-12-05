from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class ForexTradingAgent:
    def __init__(self, env):
        self.env = DummyVecEnv([lambda: env])
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./forex_trading_tensorboard/"
        )
    
    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)
        
    def save(self, path):
        self.model.save(path)
        
    def load(self, path):
        self.model = PPO.load(path, env=self.env)
