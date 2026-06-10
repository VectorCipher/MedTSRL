import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()  # Use Sigmoid to ensure output weight w is in [0, 1]
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        # Using a fixed covariance for exploration stability
        cov_matrix = torch.diag(torch.full((action_mean.size(-1),), 0.1)).to(action_mean.device)
        # Repeat for batch size
        cov_matrix = cov_matrix.unsqueeze(0).repeat(action_mean.size(0), 1, 1)
        dist = MultivariateNormal(action_mean, covariance_matrix=cov_matrix)

        action = dist.sample()
        
        # Clamp action to [0,1] just in case the normal distribution samples outside bounds
        action = torch.clamp(action, 0.0, 1.0)
        
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = torch.full((action_mean.shape[-1],), 0.01).to(action_mean.device)
        cov_mat = torch.diag(action_var).unsqueeze(0).repeat(action_mean.shape[0], 1, 1)
        
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class TutorPPO:
    def __init__(self, state_dim, action_dim, lr_actor=0.0001, lr_critic=0.0003, gamma=0.99, epochs=8, eps_clip=0.2, mini_batch_size=64):
        self.gamma = gamma
        self.epochs = epochs
        self.eps_clip = eps_clip
        self.mini_batch_size = mini_batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def to(self, device):
        self.device = device
        self.policy.to(self.device)
        self.policy_old.to(self.device)
        self.MseLoss.to(self.device) 
        return self

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = state.to(self.device)
            action, action_logprob, _ = self.policy_old.act(state_tensor)
        
        self.buffer.states.append(state_tensor.cpu())
        self.buffer.actions.append(action.cpu())
        self.buffer.logprobs.append(action_logprob.cpu())

        return action.cpu().numpy().flatten(), action_logprob.cpu().numpy().flatten()
    

    def update(self):
        if len(self.buffer.rewards) == 0:
            return

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.cat(self.buffer.states, dim=0).detach()
        old_actions = torch.cat(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.cat(self.buffer.logprobs, dim=0).detach()
        
        buffer_size = old_states.shape[0]
        
        for _ in range(self.epochs):
            indices = np.arange(buffer_size)
            np.random.shuffle(indices)

            for start in range(0, buffer_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                mb_states = old_states[mb_indices].to(self.device)
                mb_actions = old_actions[mb_indices].to(self.device)
                mb_logprobs = old_logprobs[mb_indices].to(self.device)
                mb_rewards = rewards[mb_indices].to(self.device)

                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                state_values = torch.squeeze(state_values)
                
                # Handle single element batch size squeeze bug
                if state_values.dim() == 0:
                    state_values = state_values.unsqueeze(0)

                ratios = torch.exp(logprobs - mb_logprobs.detach())

                surr1 = ratios * mb_rewards
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * mb_rewards

                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, mb_rewards) - 0.01 * dist_entropy
                
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
