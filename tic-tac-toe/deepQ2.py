import random
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Transition = namedtuple('Transition', ('S', 'A', 'S_p', 'R'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class deepQNet(nn.Module):
    
    def __init__(self, n_observations, n_actions):
         super(deepQNet, self).__init__()
         self.linear1 = nn.Linear(n_observations, 128)
         self.linear2 = nn.Linear(128,128)
         self.linear3 = nn.Linear(128, n_actions)
         

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return F.softmax(x, dim=0)
    

class DQNAgent():

    def __init__(self, alpha, gamma, eps, opt_lr, batch_size):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        
        self.batch_size = batch_size
        self.behavior_net = deepQNet(9,9).to(device)
        self.target_net = deepQNet(9,9).to(device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())

        self.optimizer = optim.Adam(self.behavior_net.parameters(), lr=opt_lr, amsgrad=True)
        self.mem = ReplayMemory(10000)

        self.steps = 0
        self.curr_state = None
        

    def move(self, state, mask):
        if not torch.any(mask):
            return None
        
        A_p  = None
        legal = [i for i in range(len(mask)) if mask[i]]
        explore = np.random.random()
        if explore < self.eps:
            A_p = np.random.choice(legal)
        else:
            with torch.no_grad():
              
               action = torch.argmax(self.behavior_net(state)[mask])
               
               j = 0
               for i in range(len(mask)):
                   if mask[i]:
                       if j == action:
                           A_p = i
                           break
                       else:
                           j += 1

        return A_p
    
    
    def update_agent(self):
        if len(self.mem) < self.batch_size:
            return
        
        transitions = self.mem.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s_p: s_p is not None, batch.S_p)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s_p for s_p in batch.S_p if s_p is not None])

        s_batch = torch.cat(batch.S)
        a_batch = torch.cat(batch.A)
        r_batch = torch.cat(batch.R)
 
        act_space = self.behavior_net(s_batch).gather(1, a_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)

        with torch.no_grad():
            next_state_values[non_final_mask] = torch.max(self.target_net(non_final_next_states),1)[0]

        pred = self.alpha * ((self.gamma * next_state_values) + r_batch)


        criterion = nn.SmoothL1Loss()
        loss = criterion(act_space, pred.unsqueeze(1))    
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.target_net.parameters(), 100)
        self.optimizer.step()

    def step_agent(self, A, S_p, R, mask, terminal, updateNet):
        A_p = None
        if not terminal:
            A_p = self.move(S_p, mask)
        
        if self.curr_state is not None:
            self.mem.push(torch.unsqueeze(self.curr_state,0), torch.tensor([A]).unsqueeze(0), torch.unsqueeze(S_p,0), torch.tensor(R,device=device).unsqueeze(0))
        self.curr_state = S_p
        self.update_agent()

        if updateNet:
            self.target_net.load_state_dict(self.behavior_net.state_dict())

        return A_p