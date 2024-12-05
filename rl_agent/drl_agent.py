from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time

from rl_env.forex_env import ForexTradingEnv

class DRLAgent:
    """Deep Reinforcement Learning Agent for Forex Trading"""
    
    def __init__(
        self,
        env: Union[ForexTradingEnv, DummyVecEnv],
        model_path: Optional[str] = None,
        train_params: Optional[Dict] = None
    ):
        """Initialize the DRL Agent
        
        Args:
            env: The forex trading environment (can be vectorized or non-vectorized)
            model_path: Path to a pretrained model (if loading existing model)
            train_params: Training parameters (if training new model)
        """
        # Check if environment is already vectorized
        if isinstance(env, DummyVecEnv):
            self.env = env
        else:
            # Wrap the environment with DummyVecEnv for Gymnasium compatibility
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
                        vf=[64, 64]
                    )
                ),
                "verbose": 0
            }
            
            # Update with custom parameters if provided
            if train_params:
                default_params.update(train_params)
            
            # Create PPO model with Gymnasium environment
            self.model = PPO(
                "MlpPolicy",
                self.env,
                **default_params
            )
    
    def train(self, total_timesteps: int = 100000) -> None:
        """Train the agent with rich progress display
        
        Args:
            total_timesteps: Total number of timesteps to train for
        """
        console = Console()
        
        # Print training info in a panel
        info_text = Text()
        info_text.append("Training Configuration\n", style="bold cyan")
        info_text.append(f"Total Timesteps: {total_timesteps:,}\n", style="green")
        info_text.append(f"Batch Size: {self.model.batch_size}\n", style="yellow")
        info_text.append(f"Learning Rate: {self.model.learning_rate}\n", style="yellow")
        info_text.append(f"Gamma: {self.model.gamma}\n", style="yellow")
        console.print(Panel(info_text, title="[bold]DRL Training", border_style="cyan"))
        
        # Calculate number of iterations
        n_steps = self.model.n_steps
        iterations = total_timesteps // n_steps
        steps_per_iter = total_timesteps // iterations
        
        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bright_green"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True,
            refresh_per_second=10  # Limit refresh rate
        ) as progress:
            # Main task for overall progress
            main_task = progress.add_task(
                "[cyan]Training Progress", 
                total=total_timesteps,
                start=True
            )
            
            start_time = time.time()
            current_timesteps = 0
            last_print_time = 0  # Track last print time
            
            for i in range(iterations):
                # Train for steps_per_iter timesteps
                self.model.learn(total_timesteps=steps_per_iter, reset_num_timesteps=False)
                current_timesteps += steps_per_iter
                
                # Update progress
                progress.update(main_task, advance=steps_per_iter)
                
                # Calculate FPS
                elapsed_time = time.time() - start_time
                fps = int(current_timesteps / elapsed_time) if elapsed_time > 0 else 0
                
                # Update metrics display (limit update frequency)
                current_time = time.time()
                if current_time - last_print_time >= 0.5:  # Update every 0.5 seconds
                    if hasattr(self.model, 'logger') and self.model.logger is not None:
                        metrics = self.model.logger.name_to_value
                        
                        # Get actual clip range values
                        if callable(self.model.clip_range):
                            clip_range = self.model.clip_range(1.0)
                        else:
                            clip_range = self.model.clip_range
                            
                        if callable(self.model.clip_range_vf):
                            clip_range_vf = self.model.clip_range_vf(1.0)
                        else:
                            clip_range_vf = self.model.clip_range_vf
                        
                        # Create metrics string
                        metrics_str = "\n=== Training Metrics ===\n"
                        metrics_str += f"FPS: {fps}\n"
                        metrics_str += f"Iterations: {i+1}\n"
                        metrics_str += f"Time Elapsed: {int(elapsed_time)}s\n"
                        metrics_str += f"Total Timesteps: {current_timesteps}\n\n"
                        
                        # Training metrics
                        metrics_str += f"Approx KL: {metrics.get('train/approx_kl', 0):.10f}\n"
                        metrics_str += f"Clip Fraction: {metrics.get('train/clip_fraction', 0):.4f}\n"
                        metrics_str += f"Clip Range: {clip_range}\n"
                        metrics_str += f"Clip Range VF: {clip_range_vf}\n"
                        metrics_str += f"Entropy Loss: {metrics.get('train/entropy_loss', 0):.3f}\n"
                        metrics_str += f"Explained Variance: {metrics.get('train/explained_variance', 0):.4f}\n"
                        metrics_str += f"Learning Rate: {self.model.learning_rate:.0e}\n"
                        metrics_str += f"Loss: {metrics.get('train/loss', 0):.1f}\n"
                        metrics_str += f"Updates: {metrics.get('train/n_updates', 0)}\n"
                        metrics_str += f"Policy Gradient Loss: {metrics.get('train/policy_gradient_loss', 0):.5f}\n"
                        metrics_str += f"Value Loss: {metrics.get('train/value_loss', 0):.1f}\n"
                        
                        # Clear screen and print metrics
                        console.clear()
                        progress.refresh()  # Redraw progress bar
                        console.print(metrics_str)
                        
                        last_print_time = current_time
                
                time.sleep(0.1)  # Small delay to make display smoother
            
            # Print completion message
            console.print("\n[bold green]Training Complete![/bold green]")
    
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
