import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch
import os
from datetime import datetime
import multiprocessing
from tqdm import tqdm

from rl_env.forex_env import ForexTradingEnv
from rl_agent.drl_agent import DRLAgent
from train_model import prepare_data

def make_env(df):
    """Create a function that returns an environment instance"""
    def _init():
        env = ForexTradingEnv(df.copy())
        return env
    return _init

def optimize_ppo(trial):
    """Optimize PPO hyperparameters."""
    # Load and prepare data
    df = prepare_data('data/raw/USDJPY_5M_2023.csv')
    
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Create vectorized environments for parallel training
    n_envs = multiprocessing.cpu_count() // 2  # Use half of available CPU cores
    env = SubprocVecEnv([make_env(df) for _ in range(n_envs)])
    
    # Suggested hyperparameters for this trial
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_int("n_steps", 1024, 4096, log=True),
        "batch_size": trial.suggest_int("batch_size", 32, 256, log=True),
        "n_epochs": trial.suggest_int("n_epochs", 5, 20),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.999),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.1, log=True),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 0.9),
        "vf_coef": trial.suggest_float("vf_coef", 0.5, 1.0),
        "device": device,
        "n_envs": n_envs
    }
    
    # Create DRL agent with trial hyperparameters
    agent = DRLAgent(env=env, train_params=hyperparams)
    
    # Create eval environment (non-vectorized for evaluation)
    eval_env = DummyVecEnv([make_env(df)])
    
    try:
        # Train the agent with progress bar
        total_timesteps = 100000
        with tqdm(total=total_timesteps, desc=f"Trial {trial.number}", unit=" steps") as pbar:
            def progress_callback(locals, globals):
                pbar.n = locals['self'].num_timesteps
                pbar.update(0)
                return True
                
            agent.model.learn(
                total_timesteps=total_timesteps,
                callback=progress_callback
            )
        
        # Evaluate the trained agent
        mean_reward = 0
        n_eval_episodes = 5
        
        for episode in tqdm(range(n_eval_episodes), desc="Evaluating", colour="yellow"):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = agent.model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                episode_reward += reward[0]  # Get first reward since env is vectorized
                
            mean_reward += episode_reward
        
        mean_reward /= n_eval_episodes
        
        # Clean up
        eval_env.close()
        env.close()
        
        return mean_reward
        
    except Exception as e:
        # Clean up
        eval_env.close()
        env.close()
        raise optuna.exceptions.TrialPruned()

def main():
    # Create rl_models directory if it doesn't exist
    os.makedirs("rl_models", exist_ok=True)
    
    # Use a single database for all optimization runs
    study_name = "ppo_optimization"
    storage_name = "sqlite:///rl_models/optimization_history.db"
    
    # Create study
    sampler = TPESampler(n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True  # Load existing study if available
    )
    
    try:
        print("\nStarting optimization...")
        print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
        print(f"Number of parallel environments: {multiprocessing.cpu_count() // 2}")
        print(f"Total trials: 50")
        print(f"Timeout: 12 hours\n")
        
        study.optimize(
            optimize_ppo,
            n_trials=50,
            timeout=3600 * 12  # 12 hours timeout
        )
        
        print("\nBest trial:")
        trial = study.best_trial
        
        print(f"Value: {trial.value}")
        print("\nBest parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
    except KeyboardInterrupt:
        print("\nOptimization stopped by user.")
        
if __name__ == "__main__":
    main()
