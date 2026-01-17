import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Common Feature Extractor
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor
        self.actor = nn.Linear(hidden_dim, output_dim)
        
        # Critic
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor: returns logits for actions
        action_logits = self.actor(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic: returns value estimate
        state_value = self.critic(x)
        
        return action_probs, state_value

class TradingAgent(pl.LightningModule):
    """
    PPO Agent implemented as a LightningModule.
    """
    def __init__(self, env: gym.Env, lr=1e-3, gamma=0.99, clip_eps=0.2):
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.clip_eps = clip_eps
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.model = ActorCritic(self.obs_dim, self.action_dim)
        
        # Buffer to store trajectories
        self.batch_states = []
        self.batch_actions = []
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_dones = []
        
        # Important: We need manual optimization for PPO usually, unless we restructure data loading.
        # But for simplicity, we'll collect data in `training_step`? 
        # Actually PPO typically collects a full rollout then updates.
        # Lightning usually expects "batch" input in training_step.
        
        # We will use an "IterableDataset" style approach or just run the environment loop inside the training step
        # effectively doing "one rollout per step" as a pseudo-batch.
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def training_step(self, batch, batch_idx):
        # NOTE: 'batch' is ignored primarily because we generate our own data from the environment.
        # In a real rigorous setup, you'd use a DataLoader that yields rollouts.
        
        optimizer = self.optimizers()
        
        # 1. Collect Rollout
        state, _ = self.env.reset()
        done = False
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        
        # Simulate N steps or until done (Truncated for efficiency)
        rollout_steps = 200 
        for _ in range(rollout_steps):
            state_tensor = torch.FloatTensor(state).to(self.device)
            probs, value = self.model(state_tensor.unsqueeze(0))
            m = Categorical(probs)
            action = m.sample()
            
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            
            states.append(state_tensor)
            actions.append(action)
            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            
            state = next_state
            if done:
                break
                
        # 2. Compute Advantages (GAE or simple monte carlo)
        # Simple Monte Carlo for now
        returns = []
        R = 0
        # If not done, bootstrap from value of last state
        if not done:
            with torch.no_grad():
                _, next_val = self.model(torch.FloatTensor(state).to(self.device).unsqueeze(0))
                R = next_val.item()
                
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7) # Normalize
        
        # 3. PPO Update
        # Stack
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs).detach()
        values = torch.cat(values).squeeze()
        
        # Calculate advantage
        # adv = returns - values.detach() # Standard Advantage
        
        # Re-evaluate
        probs, new_values = self.model(states)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratios = torch.exp(new_log_probs - old_log_probs.squeeze())
        
        # Surrogate Loss
        advantages = returns - values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(new_values.squeeze(), returns)
        
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("reward", sum(rewards), prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        # Dummy dataloader just to satisfy Lightning
        # In a real app we might wrap the environment in an IterableDataset
        return torch.utils.data.DataLoader([0]*1000) # 1000 pseudo-epochs

if __name__ == "__main__":
    # Test Instantiation
    # Dummy env mock
    # env = ...
    # model = TradingAgent(env)
    pass
