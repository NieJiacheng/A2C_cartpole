import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical



class Actor(nn.Module):
    '''
    演员Actor网络
    '''
    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, action_dim)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s[0], np.ndarray):
            s = torch.FloatTensor(s[0])
        elif isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = F.softmax(self.fc2(x), dim=-1)

        return out


class Critic(nn.Module):
    '''
    评论家Critic网络
    '''
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 1)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s[0], np.ndarray):
            s = torch.FloatTensor(s[0])
        elif isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = self.fc2(x)

        return out


class Actor_Critic:
    def __init__(self, env, lr_a, lr_c):
        self.gamma = 0.99
        self.lr_a = lr_a
        self.lr_c = lr_c

        self.env = env
        self.action_dim = self.env.action_space.n             #获取描述行动的数据维度
        self.state_dim = self.env.observation_space.shape[0]  #获取描述环境的数据维度

        self.actor = Actor(self.action_dim, self.state_dim)   #创建演员网络
        self.critic = Critic(self.state_dim)                  #创建评论家网络

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.loss = nn.MSELoss()

    def get_action(self, s):
        a = self.actor(s)
        dist = Categorical(a)
        action = dist.sample()             #可采取的action
        log_prob = dist.log_prob(action)   #每种action的概率

        return action.detach().numpy(), log_prob

    def learn(self, log_prob_s, s_s, done, rew_s, entropy_coef):
        #使用Critic网络估计状态值
        if done:
            v_ = torch.tensor([0])
        else:
            v_ = self.critic(s_s.pop()).squeeze()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        critic_loss = torch.tensor([0.])
        loss_actor = torch.tensor([0.])
        for i in range(len(s_s)):
            s = s_s[len(s_s) - 1 - i]
            rew = rew_s[len(rew_s) - 1- i]
            log_prob = log_prob_s[len(log_prob_s) - 1 - i]
            v = self.critic(s).squeeze()
            v_ = self.gamma * v_ + rew
            critic_loss += self.loss(v_, v)
            advantage = v_ - v
            entropy = -torch.sum(self.actor(s) * torch.log(self.actor(s)))
            loss_actor += -log_prob * advantage.detach() - entropy_coef * entropy
        critic_loss.backward()
        self.critic_optim.step()
        loss_actor.backward()
        self.actor_optim.step()
