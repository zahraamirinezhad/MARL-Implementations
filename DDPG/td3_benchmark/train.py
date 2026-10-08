import os
import yaml
import numpy as np
import torch as T
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
from torch.utils.tensorboard import SummaryWriter
from src.agents.td3_agent import TD3Agent
from src.utils.metrics import compute_iqm, stratified_bootstrap_ci

def evaluate_policy(agent, env, eval_episodes=5):
    eval_rewards = []
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        score = 0.0
        while not done:
            action = agent.choose_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        eval_rewards.append(score)
    return np.mean(eval_rewards)

def save_gif(agent, env_id, filepath="lunarlander_td3.gif", fps=30):
    env = gym.make(env_id, render_mode="rgb_array")
    images = []
    obs, _ = env.reset()
    done = False
    
    while not done:
        frame = env.render()
        images.append(frame)
        action = agent.choose_action(obs, evaluate=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    env.close()
    imageio.mimsave(filepath, images, fps=fps)
    print(f"\nGIF saved successfully at {filepath}")

def plot_learning_curve(episodes, train_scores, eval_episodes, eval_scores, save_path="learning_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, train_scores, label="Train Reward (Noisy)", alpha=0.4, color="gray")
    
    # Moving Average for training
    window = 10
    if len(train_scores) >= window:
        smoothed = np.convolve(train_scores, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(train_scores)), smoothed, label=f"Train Moving Avg ({window})", color="blue")

    plt.plot(eval_episodes, eval_scores, label="Eval Reward (Deterministic)", color="red", linewidth=2, marker='o')
    plt.title("TD3 Learning Curve - LunarLanderContinuous-v3")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.axhline(y=200, color='g', linestyle='--', label='Solved Threshold (200)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Learning curve plot saved at {save_path}")

def run_experiment(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device} | Env: {cfg['env_id']}")

    env = gym.make(cfg['env_id'])
    eval_env = gym.make(cfg['env_id'])

    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    agent = TD3Agent(cfg, input_dims, n_actions, max_action, min_action, device)
    writer = SummaryWriter(f"runs/{cfg['algo']}_{cfg['env_id']}_seed_{cfg['seed']}")

    train_scores = []
    eval_scores = []
    eval_episodes_list = []

    for i in range(cfg['n_games']):
        obs, _ = env.reset(seed=cfg['seed'] + i)
        done = False
        score = 0.0

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(obs, action, reward, obs_, done)
            actor_loss, critic_loss = agent.learn()

            score += reward
            obs = obs_

        train_scores.append(score)

        # Evaluation Protocol (Isolated from Noise)
        if (i + 1) % cfg['eval_interval'] == 0:
            eval_score = evaluate_policy(agent, eval_env, cfg['eval_episodes'])
            eval_scores.append(eval_score)
            eval_episodes_list.append(i + 1)
            writer.add_scalar('Evaluation/Deterministic_Reward', eval_score, i)
            print(f"Episode {i+1:03d}/{cfg['n_games']} | Train Score: {score:.1f} | Eval Score (No Noise): {eval_score:.1f}")

    env.close()
    eval_env.close()
    writer.close()

    # IQM and Bootstrap Calculations
    eval_array = np.array(eval_scores)
    iqm_val = compute_iqm(eval_array)
    ci_low, ci_high = stratified_bootstrap_ci(eval_array)

    print("\n--- Final Scientific Evaluation Results ---")
    print(f"Interquartile Mean (IQM): {iqm_val:.2f}")
    print(f"95% Stratified Bootstrap CI: [{ci_low:.2f}, {ci_high:.2f}]")

    # Plot & Save Artifacts
    os.makedirs("results", exist_ok=True)
    plot_learning_curve(range(1, cfg['n_games'] + 1), train_scores, eval_episodes_list, eval_scores, "results/learning_curve.png")
    save_gif(agent, cfg['env_id'], "results/lunarlander_agent.gif")

if __name__ == "__main__":
    run_experiment("td3_benchmark/configs/env_lunarlander.yaml")
