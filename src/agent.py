"""
Signal.Engine - PPO Agent
Enhanced TradingAgent with better architecture for GPU utilization.
"""
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic network with improved architecture.
    Larger model for better GPU utilization.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Common Feature Extractor (deeper for more GPU work)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization for training stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction with residual-like connections
        x = F.relu(self.ln1(self.fc1(x)))
        identity = x
        x = F.relu(self.ln2(self.fc2(x)))
        x = x + identity  # Skip connection
        x = F.relu(self.ln3(self.fc3(x)))
        
        # Actor output
        action_logits = self.actor(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic output
        state_value = self.critic(x)
        
        return action_probs, state_value
    
    def get_action_and_value(self, x):
        """Combined forward for efficiency."""
        probs, value = self.forward(x)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class TradingAgent(pl.LightningModule):
    """
    PPO Agent implemented as a LightningModule.
    Enhanced for better GPU utilization.
    """
    def __init__(self, env: gym.Env, lr=3e-4, gamma=0.99, clip_eps=0.2, 
                 gae_lambda=0.95, rollout_steps=512, ppo_epochs=4,
                 value_coef=0.5, entropy_coef=0.01):
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.model = ActorCritic(self.obs_dim, self.action_dim, hidden_dim=256)
        
        # Disable automatic optimization for PPO's multi-epoch updates
        self.automatic_optimization = False
        
        # Save hyperparameters (except env which isn't serializable)
        self.save_hyperparameters(ignore=['env'])
    
    def on_train_epoch_start(self):
        """
        Entropy Scheduling: Decay entropy coefficient linearly/exponentially.
        Start high (Exploration) -> End low (Exploitation).
        """
        # Decay factor per epoch (e.g., 0.95)
        decay = 0.95
        new_entropy = max(0.001, self.hparams.entropy_coef * (decay ** self.current_epoch))
        
        # Update the active entropy coef (not the hparams one, dev convention)
        self.entropy_coef = new_entropy
        self.log("entropy_coef", self.entropy_coef, prog_bar=True)
        
    def forward(self, x):
        return self.model(x)
    
    def select_action(self, state):
        """Select action for inference."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, value = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        
        # 1. Collect Rollout
        state, _ = self.env.reset()
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        for _ in range(self.rollout_steps):
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            with torch.no_grad():
                probs, value = self.model(state_tensor.unsqueeze(0))
            
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            
            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.squeeze())
            dones.append(float(done))
            
            state = next_state
            if done:
                state, _ = self.env.reset()
        
        # 2. Compute GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            _, next_value = self.model(torch.FloatTensor(state).to(self.device).unsqueeze(0))
            next_value = next_value.squeeze()
        
        # Stack tensors
        states = torch.stack(states)
        actions = torch.stack(actions).squeeze()
        old_log_probs = torch.stack(log_probs).squeeze().detach()
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.stack(values).squeeze()
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # GAE calculation
        advantages = torch.zeros_like(rewards_tensor)
        returns = torch.zeros_like(rewards_tensor)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
                next_non_terminal = 1.0 - dones_tensor[t]
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - dones_tensor[t]
            
            delta = rewards_tensor[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 3. PPO Update (multiple epochs)
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.ppo_epochs):
            # Full batch update (can be changed to mini-batches for larger rollouts)
            probs, new_values = self.model(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values.squeeze(), returns)
            
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            optimizer.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            n_updates += 1
        
        # Logging
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        total_reward = sum(rewards)
        
        self.log("train_loss", total_loss / n_updates, prog_bar=True)
        self.log("actor_loss", total_actor_loss / n_updates)
        self.log("critic_loss", total_critic_loss / n_updates)
        self.log("entropy", total_entropy / n_updates)
        self.log("reward", total_reward, prog_bar=True)
        self.log("avg_reward", avg_reward)
        
        return torch.tensor(total_loss / n_updates)
    
    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
    
    def train_dataloader(self):
        # Dummy dataloader to satisfy Lightning's requirement
        # Actual data comes from environment rollouts
        return torch.utils.data.DataLoader(
            torch.zeros(500),  # 500 training steps per epoch
            batch_size=1,
            num_workers=0
        )


if __name__ == "__main__":
    # Test model parameter count
    model = ActorCritic(input_dim=10, output_dim=3, hidden_dim=256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")  # Should be ~200K-300K

