
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical

from drl.agents.agent import Agent

EPS = np.finfo(np.float32).eps.item()
torch.manual_seed(543)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class REINFORCE(Agent):
    def __init__(self, env, learning_rate=1e-2, gamma=0.99):
        super().__init__()
        self.policy: Policy = Policy()
        self.env = env
        self.ep_reward = 0.0
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma

    def predict(self, observation):
        state = torch.from_numpy(observation).float().unsqueeze(0)
        probs = self.policy.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def reset(self):
        self.ep_reward = 0

    def update_post_episode(self):
        discounted_reward = 0  # G
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)
        if not np.isnan(returns.std().numpy()):
            returns = (returns - returns.mean()) / (returns.std() + EPS)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss_sum = torch.cat(policy_loss).sum()
        policy_loss_sum.backward()
        self.optimizer.step()
        self.policy.rewards = []
        self.policy.saved_log_probs = []

    def update(self, observation, reward, done, info):
        self.env.render()
        self.policy.rewards.append(reward)
        self.ep_reward += reward
        if done:
            self.update_post_episode()

        return self.ep_reward
