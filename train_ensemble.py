import optuna
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import os
import multiprocessing
from tqdm import tqdm
from datetime import datetime
import json

from rl_env.forex_env import ForexTradingEnv
from rl_agent.drl_agent import DRLAgent
from train_model import prepare_data
from optimize_model import make_env

def load_best_params():
    """Load best parameters from optimization database"""
    storage_name = "sqlite:///rl_models/optimization_history.db"
    study = optuna.load_study(study_name="ppo_optimization", storage=storage_name)
    return study.best_trial.params

def evaluate_model(agent, env, n_episodes=100):
    """Evaluate a model over multiple episodes"""
    rewards = []
    for _ in tqdm(range(n_episodes), desc="Evaluating", unit=" episodes"):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards)

def train_model(params, model_number, total_timesteps=1_000_000):
    """Train a single model with given parameters"""
    # Load and prepare data
    df = prepare_data('data/raw/USDJPY_5M_2023.csv')
    
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Create vectorized environments
    n_envs = multiprocessing.cpu_count() // 2
    env = SubprocVecEnv([make_env(df) for _ in range(n_envs)])
    
    # Add device and n_envs to parameters
    params = params.copy()
    params.update({
        "device": device,
        "n_envs": n_envs
    })
    
    # Create agent
    agent = DRLAgent(env=env, train_params=params)
    
    # Train with progress bar
    with tqdm(total=total_timesteps, desc=f"Training Model {model_number}", unit=" steps") as pbar:
        def progress_callback(locals, globals):
            pbar.n = locals['self'].num_timesteps
            pbar.update(0)
            return True
            
        agent.model.learn(
            total_timesteps=total_timesteps,
            callback=progress_callback
        )
    
    # Evaluate
    eval_env = SubprocVecEnv([make_env(df)])
    mean_reward, std_reward = evaluate_model(agent, eval_env)
    
    # Save model
    timestamp = int(datetime.timestamp(datetime.now()))
    save_path = f"rl_models/ensemble/model_{model_number}_{timestamp}"
    agent.model.save(save_path)
    
    # Clean up
    env.close()
    eval_env.close()
    
    return {
        "model_number": model_number,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "model_path": save_path
    }

def main():
    # Create directories
    os.makedirs("rl_models/ensemble", exist_ok=True)
    
    # Load best parameters
    print("Loading best parameters from optimization...")
    best_params = load_best_params()
    
    # Settings
    n_models = 5  # Number of models to train
    timesteps_per_model = 1_000_000  # Training steps per model
    
    print("\nStarting ensemble training...")
    print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"Number of parallel environments: {multiprocessing.cpu_count() // 2}")
    print(f"Number of models: {n_models}")
    print(f"Steps per model: {timesteps_per_model:,}\n")
    
    # Train models
    results = []
    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models}")
        result = train_model(best_params, i+1, timesteps_per_model)
        results.append(result)
        
        print(f"Model {i+1} Results:")
        print(f"Mean reward: {result['mean_reward']:.2f}")
        print(f"Std reward: {result['std_reward']:.2f}")
        print(f"Saved to: {result['model_path']}")
    
    # Save ensemble results
    timestamp = int(datetime.timestamp(datetime.now()))
    results_path = f"rl_models/ensemble/results_{timestamp}.json"
    
    ensemble_results = {
        "parameters": best_params,
        "models": results,
        "ensemble_stats": {
            "mean_reward": float(np.mean([r['mean_reward'] for r in results])),
            "std_reward": float(np.std([r['mean_reward'] for r in results]))
        }
    }
    
    with open(results_path, "w") as f:
        json.dump(ensemble_results, f, indent=4)
    
    print(f"\nEnsemble Results:")
    print(f"Mean reward across models: {ensemble_results['ensemble_stats']['mean_reward']:.2f}")
    print(f"Std reward across models: {ensemble_results['ensemble_stats']['std_reward']:.2f}")
    print(f"Full results saved to: {results_path}")

if __name__ == "__main__":
    main()
