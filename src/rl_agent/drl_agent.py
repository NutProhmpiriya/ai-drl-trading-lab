from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn

from rl_env.forex_env import ForexTradingEnv

class DRLAgent:
    """Deep Reinforcement Learning Agent for Forex Trading"""
    
    def __init__(
        self,
        env: ForexTradingEnv,
        model_path: Optional[str] = None,
        train_params: Optional[Dict] = None
    ):
        """Initialize the DRL Agent
        
        Args:
            env: The forex trading environment
            model_path: Path to a pretrained model (if loading existing model)
            train_params: Training parameters (if training new model)
        """
        self.env = DummyVecEnv([lambda: env])
        
        if model_path:
            self.model = PPO.load(model_path, env=self.env)
        else:
            # Default training parameters
            default_params = {
                "learning_rate": 0.00003,
                "n_steps": 1024,
                "batch_size": 32,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "clip_range_vf": 0.2,
                "normalize_advantage": True,
                "ent_coef": 0.05,
                "vf_coef": 1.0,
                "max_grad_norm": 0.5,
                "use_sde": False,
                "sde_sample_freq": -1,
                "target_kl": None,
                "tensorboard_log": "./tensorboard_log",
                "policy_kwargs": dict(
                    net_arch=dict(
                        pi=[64, 64],
                        vf=[256, 128, 64]
                    ),
                    activation_fn=nn.ReLU
                ),
                "verbose": 1
            }
            
            # Update with custom parameters if provided
            if train_params:
                default_params.update(train_params)
            
            self.model = PPO("MlpPolicy", self.env, **default_params)
    
    def train(self, total_timesteps: int = 100000) -> None:
        """Train the agent
        
        Args:
            total_timesteps: Total number of timesteps to train for
        """
        self.model.learn(total_timesteps=total_timesteps)
    
    def save(self, save_path: str) -> None:
        """Save the trained model
        
        Args:
            save_path: Path to save the model to
        """
        self.model.save(save_path)
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Make a prediction for a given observation
        
        Args:
            observation: The current environment observation
            deterministic: Whether to use deterministic actions
        
        Returns:
            action: The predicted action
            _states: Internal agent states (if any)
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def evaluate(self, num_episodes: int = 1) -> List[float]:
        """Evaluate the agent's performance
        
        Args:
            num_episodes: Number of episodes to evaluate over
        
        Returns:
            episode_rewards: List of total rewards for each episode
        """
        episode_rewards = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _states = self.predict(obs)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward[0]  # Unwrap from vectorized environment
            
            episode_rewards.append(total_reward)
        
        return episode_rewards
