import numpy as np
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Categorical

# Neural Network Policy
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        x = self.network(x)
        x = nn.Softmax(dim=-1)(x)
        return x

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns

# Vanilla Policy Gradient train loop from scratch

def train_vpg(env, policy, num_episodes, gamma=0.99, lr=0.0005):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rewards_history = []
    loss_history = []
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        episode_reward = 0
        state = torch.from_numpy(state[0]).float().unsqueeze(0)
        done = False
        while not done:
            action_prob = policy(state)
            action_dist = Categorical(action_prob)
            action = action_dist.sample()
            next_state, reward, terminated, truncated, info = env.step(action.item())
            episode_reward += reward
            done = terminated or truncated
            log_probs.append(action_dist.log_prob(action))
            next_state = torch.tensor(next_state, dtype=torch.float32)
            state = next_state
            rewards.append(reward)
            if done:
                break
        discounted_rewards = compute_returns(rewards, gamma)
        policy_loss_sum = 0
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss_sum += -log_prob * reward
        optimizer.zero_grad()
        policy_loss_sum.backward()
        optimizer.step()
        rewards_history.append(episode_reward)
        loss_history.append(policy_loss_sum.item())

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Loss: {loss_history[-1]:.4f}")
        if np.mean(rewards_history[-100:]) >= 400.0:
            print(f"Solved in {episode + 1} episodes!")
            break
        return rewards_history, loss_history

def plot_results(rewards_history, loss_history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(rewards_history)
    ax1.set_title('Rewards per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    ax2.plot(loss_history)
    ax2.set_title('Loss per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

# Call this function after training
plot_results(rewards_history, loss_history)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = Policy(state_dim, action_dim)
    rewards_history, loss_history = train_vpg(env, policy, num_episodes=1000)
    plot_results(rewards_history, loss_history)

        
        
