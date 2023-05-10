import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from colors import colors

has_cuda = torch.cuda.is_available()
has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
if has_cuda: device = torch.device('cuda')
elif has_mps: device = torch.device('mps')
else: device = torch.device('cpu')

# TODO: reconsider the network structure
class GeneralNet(nn.Module):
    """
    state image feature extraction
    """
    def __init__(self, in_ch = 1) -> None:
        super(GeneralNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.avg_pool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        output = self.conv(input) # [1, in_ch, 84, 84] -> [1, 256, 3, 3]
        output = self.avg_pool(output) # [1, 256, 3, 3] -> [1, 256, 1, 1] # TODO
        return output


class ActorPolicy(nn.Module):
    def __init__(self, action_dim = 3) -> None:
        super(ActorPolicy, self).__init__()
        self.general_net = GeneralNet()
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, state):
        output = self.general_net(state)
        output = self.fcs(output)
        mean = self.tanh(output)
        std = self.softplus(output)

        return mean, std
    

class CriticValue(nn.Module):
    def __init__(self) -> None:
        super(CriticValue, self).__init__()
        self.general_net = GeneralNet()
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        output = self.general_net(state)
        value = self.fcs(output)

        return value
    

class Record():
    def __init__(self, state, action, a_logp, reward, next_state) -> None:
        self.state = state
        self.action = action
        self.a_logp = a_logp
        self.reward = reward
        self.next_state = next_state

class Memory():
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.a_logps = []
        self.rewards = []
        self.next_states = []
    
    def add(self, record: Record):
        self.states.append(record.state)
        self.actions.append(record.action)
        self.a_logps.append(record.a_logp)
        self.rewards.append(record.reward)
        self.next_states.append(record.next_state)

    def clear(self):
        self.states = []
        self.actions = []
        self.a_logps = []
        self.rewards = []
        self.next_states = []


class PPOAgent():
    def __init__(self, 
                action_dim = 1,
                gamma = 0.9,
                ppo_epoch_num = 10,
                batch_size = 8,
                memory_cap = 2000,
                clip_epsilon = 0.2,
                learning_rate = 0.0005
                ) -> None:
        super(PPOAgent, self).__init__()

        self.actor = ActorPolicy().to(device)
        self.critic = CriticValue().to(device)

        # TODO; lr and weight_decay?
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.cnt = 0
        # NOTE: hyperparameters
        self.gamma = gamma
        self.ppo_epoch_num = ppo_epoch_num
        self.batch_size = batch_size
        self.memory_cap = memory_cap
        self.clip_epsilon = clip_epsilon

        self.memory = Memory()

        # NOTE: for result saving
        self.td_errs_mean = []
        self.td_errs_std = []
        self.actor_loss_list = []
        self.critic_loss_list = []
        

    def get_action(self, state):
        with torch.no_grad():
            mean, std = self.actor(state.to(device))
            mean, std = mean.squeeze(0), std.squeeze(0)
            mean = torch.nan_to_num(mean); std = torch.nan_to_num(std)
        action_dist = torch.distributions.Normal(mean, std+1e-20)
        action = action_dist.sample()
        a_logp = action_dist.log_prob(action)

        action = action.cpu(); a_logp = a_logp.cpu()

        action = action.tolist()
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0.8, 1)
        action[2] = np.clip(action[2], 0, 0.2)

        # action = [action[0]*1.5, action[1]+1, action[1]+1]
        # print(colors.OKCYAN, "\naction", action, a_logp, colors.ENDC)

        return tuple(action), a_logp # TODO: type?
    
    def store(self, record):
        if self.cnt == self.memory_cap:
            self.cnt = 0
            self.memory.clear()

        self.memory.add(record)
        self.cnt += 1

        return self.cnt % self.memory_cap == 0
    
    def save_model(self, epi_num):
        torch.save(self.actor.state_dict(), 'models/4new_actor_' + str(epi_num) +'.pkl')
        torch.save(self.critic.state_dict(), 'models/4new_critic_' + str(epi_num) + '.pkl')

    def save_loss(self, reward_list: list):
        t_dict = {
            'rewards': reward_list,
            'td_errs_mean': self.td_errs_mean,
            'td_errs_std': self.td_errs_std,
            'a_loss': self.actor_loss_list,
            'c_loss': self.critic_loss_list
        }

        for k, v in t_dict.items():
            (pd.DataFrame(v)).to_csv('losses/loss_' + k + '.csv', index=False)

    
    def learn(self):
        states = torch.stack(self.memory.states).squeeze(1).to(device)
        actions = torch.FloatTensor(self.memory.actions).to(device)
        rewards = torch.FloatTensor(self.memory.rewards).view(-1, 1).to(device)
        next_states = torch.stack(self.memory.next_states).squeeze(1).to(device)
        old_action_logps = torch.stack(self.memory.a_logps).to(device)
        # old_action_logps = self.memory.a_logps
        # print(colors.OKCYAN, "\n", actions, actions.shape, next_states.shape, colors.ENDC)

        # NOTE: calculate td-error
        with torch.no_grad():
            next_s_target = self.critic(next_states)
            cur_s_target = self.critic(states)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-20)
        td_q = rewards + self.gamma * next_s_target
        td_err = td_q - cur_s_target

        tm, ts = torch.std_mean(td_err.cpu())
        print(colors.FAIL, "\ntd_err mean std ", tm, ts, colors.ENDC)
        self.td_errs_mean.append(tm.item())
        self.td_errs_std.append(ts.item())

        for _ in range(self.ppo_epoch_num):
            for i in BatchSampler(SubsetRandomSampler(range(self.memory_cap)), self.batch_size, False):
                # print("\n\n ", i, " \n\n")
                mean, std = self.actor(states[i])
                mean = torch.nan_to_num(mean); std = torch.nan_to_num(std)
                action_dist = torch.distributions.Normal(mean, std+1e-20)
                action_logp = action_dist.log_prob(actions[i])
                ratio = torch.exp(action_logp - old_action_logps[i])

                surr1 = ratio * td_err[i]
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * td_err[i]
                surr1 = torch.nan_to_num(surr1); surr2 = torch.nan_to_num(surr2)
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (F.mse_loss(self.critic(states[i]), td_q[i])).mean()

                # print(colors.FAIL, "\nactor loss:", actor_loss, colors.ENDC)
                # print(colors.FAIL, "\ncritic loss:", critic_loss, colors.ENDC)
                self.actor_loss_list.append(actor_loss.cpu().item())
                self.critic_loss_list.append(critic_loss.cpu().item())

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
