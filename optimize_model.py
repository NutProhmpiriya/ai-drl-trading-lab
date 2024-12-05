import optuna
from optuna.pruners import NopPruner
from optuna.samplers import TPESampler
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import os
from datetime import datetime
import multiprocessing
import json
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

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
    
    # Force using CPU for better performance with MlpPolicy
    device = "cpu"
    print(f"Using device: {device}")
    
    # Create vectorized environments for parallel training
    n_envs = multiprocessing.cpu_count() // 2  # Use half of available CPU cores
    print(f"Number of parallel environments: {n_envs}")
    envs = [make_env(df) for _ in range(n_envs)]
    env = DummyVecEnv(envs)
    
    # Suggested hyperparameters for this trial
    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_int("n_steps", 1024, 4096, log=True),
        "batch_size": trial.suggest_int("batch_size", 32, 256),
        "n_epochs": trial.suggest_int("n_epochs", 5, 20),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.999),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.1, log=True),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 0.9),
        "vf_coef": trial.suggest_float("vf_coef", 0.5, 1.0),
        "device": device
    }
    
    # Adjust batch_size to be a factor of n_steps * n_envs
    buffer_size = hyperparams["n_steps"] * n_envs
    valid_batch_sizes = [i for i in range(32, min(257, buffer_size + 1)) if buffer_size % i == 0]
    if not valid_batch_sizes:
        raise optuna.exceptions.TrialPruned()
    hyperparams["batch_size"] = min(valid_batch_sizes, key=lambda x: abs(x - hyperparams["batch_size"]))
    
    # Create DRL agent with trial hyperparameters
    agent = DRLAgent(env=env, train_params=hyperparams)
    
    # Create eval environment (non-vectorized for evaluation)
    eval_env = DummyVecEnv([make_env(df)])
    
    try:
        # Train the agent with progress bar
        total_timesteps = 100000
        with tqdm(total=total_timesteps, desc=f"Trial {trial.number} Training") as pbar:
            def callback(locals, globals):
                pbar.n = locals['self'].num_timesteps
                pbar.update(0)
                return True
            agent.model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Evaluate the agent
        n_eval_episodes = 5
        mean_reward = 0
        
        for episode in tqdm(range(n_eval_episodes), desc=f"Trial {trial.number} Evaluation"):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = agent.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
            mean_reward += episode_reward
            tqdm.write(f"Episode {episode + 1} reward: {episode_reward:.2f}")
        
        mean_reward = mean_reward / n_eval_episodes
        tqdm.write(f"Trial {trial.number} mean reward: {mean_reward:.2f}")
        
        return mean_reward
        
    except Exception as e:
        tqdm.write(f"Error in trial {trial.number}: {str(e)}")
        raise optuna.exceptions.TrialPruned()

def main():
    """Main optimization function."""
    print("\nStarting optimization...")
    
    # Create study in SQLite database
    storage_name = "sqlite:///study.db"
    study_name = "forex_trading_optimization"

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            sampler=TPESampler(),
            pruner=optuna.pruners.NopPruner(),  # ใช้ NopPruner เพื่อไม่ให้มีการ prune
            load_if_exists=True
        )
        
        if len(study.trials) > 0:
            print(f"\nLoaded existing study with {len(study.trials)} trials")
            print(f"Best value so far: {study.best_value:.3f}")
        else:
            print("\nCreated new study")
            
    except Exception as e:
        print(f"\nError with storage: {str(e)}")
        print("Creating in-memory study instead...")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(),
            pruner=optuna.pruners.NopPruner()  # ใช้ NopPruner เพื่อไม่ให้มีการ prune
        )
    
    # Print optimization settings
    device = "cpu"
    n_envs = multiprocessing.cpu_count() // 2
    N_TRIALS = 50
    TIMEOUT = 12
    print(f"\nOptimization settings:")
    print(f"Device: {device}")
    print(f"Number of parallel environments: {n_envs}")
    print(f"Total trials: {N_TRIALS}")
    print(f"Timeout: {TIMEOUT} hours\n")
    
    try:
        study.optimize(
            optimize_ppo,
            n_trials=N_TRIALS,
            timeout=3600 * TIMEOUT,  # Convert hours to seconds
            catch=(Exception,),  # Catch all exceptions to prevent study from crashing
            show_progress_bar=True
        )
        
        print("\nOptimization finished!")
        print(f"Best trial value: {study.best_value:.3f}")
        print(f"Best trial params:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
        if isinstance(study.storage, optuna.storages.RDBStorage):
            print(f"\nTo view the optimization dashboard, run:")
            print(f"optuna-dashboard {storage_name}")
        
    except KeyboardInterrupt:
        print("\nOptimization stopped by user.")
    except Exception as e:
        print(f"\nOptimization failed with error: {str(e)}")
    
    # Try to save best parameters even if optimization failed
    try:
        if len(study.trials) > 0:
            print("\nBest parameters found:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
            if isinstance(study.storage, optuna.storages.RDBStorage):
                print(f"\nTo view the optimization dashboard, run:")
                print(f"optuna-dashboard {storage_name}")
    except Exception as e:
        print(f"\nCould not retrieve best parameters: {str(e)}")

if __name__ == "__main__":
    main()
