"""
The module implements Double Q-learning algorithm, as per (Hasselt et al., 2015).
The code is an adapted version of PyTorch's DQN tutorial: 
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""




import random
from collections import namedtuple, deque
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'

# The object used for storing the update transitions, represents a tuple containing:
# current state, action taken, next state and reward obtained for the taken action
Transition = namedtuple('Transition', ('S', 'A', 'S_p', 'R'))


class ReplayMemory(object):
    """ 
    The implementation of Experience Replay buffer which stores the update transition. The default size is 10000 transitions
    """
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class deepQNet(nn.Module):
    """ 
    The Network architecture used for training. The output layer is a linear function with no activation applied on it, so that
    the Q-value updates can observe a full range of outputs.
    """
    def __init__(self, n_observations, n_actions):
        super(deepQNet, self).__init__()
        self.linear1 = nn.Linear(n_observations, 128)
        self.linear2 = nn.Linear(128,128)
        self.linear3 = nn.Linear(128,128)
        self.linear4 = nn.Linear(128, n_actions)


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x

    

class DQNAgent():
    """
    The class implements the agent that runs the algorithm.
    The agent maintains two networks: policy network (behavior_net) and target network (target_net).
    Policy network is used for action selection during the training.
    Target network is updated gradually by soft updates - 
    """

    def __init__(self, in_obs, out_actions, alpha, gamma, eps_start, eps_end, eps_decay, opt_lr, batch_size, tau, softUpdates=True):
        """Initialization of the agent. 

        Args:
            in_obs (long): number of observations considered within one state (number of input units of the networks)
            out_actions (long): number of possible actions that the agent can take (number of output units of the networks)
            alpha (float): the learning rate used for the Q-value updates
            gamma (float): the discount factor
            eps_start (float): the starting value of the exploration parameter
            eps_end (float): the final value of the exploration parameter
            eps_decay (long): number of actions taken until the exploration rate reaches eps_end
            opt_lr (float): the starting learning rate of the optimizer(Adam)
            batch_size (long): number of transitions used for the networks' update
            tau (float): the rate at which target network gets updates
            softUpdates (bool, optional): the type of target network updates. If set to false, hard updates every *tau*
            actions taken are performed. Defaults to True.
        """
        # initialize the parameters
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.tau = tau
        self.softUpdates = softUpdates


        # initialize the networks, as well as the buffer, optimizer and scheduler which adapts the learning rate of the optimizer
        self.behavior_net = deepQNet(in_obs, out_actions).to(device)
        self.target_net = deepQNet(in_obs, out_actions).to(device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())

        self.optimizer = optim.Adam(self.behavior_net.parameters(), lr=opt_lr, amsgrad=True)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5000, factor=0.1, verbose=True, min_lr=1e-10)
        self.scheduler = optim.lr_scheduler.ConstantLR(self.optimizer, factor=0.1, total_iters=8)
        self.mem = ReplayMemory(50000)

        # variable tracking number of actions taken
        self.steps = 0

        # list which tracks the loss progression
        self.losses = []
        
        # variables holding the current state (the agent learns via self-play and performs actions for both players
        # which is why we need two state-holders, one for each player)
        self.curr_state_p0 = None
        self.curr_state_p1 = None

    def move(self, state, mask):
        """
        The function  which is responsible for choosing the action, given the current state,
        as well as the mask.

        Args:
            state (tensor of size equal to the no. of input units of the networks): tensor representing the current state of the environment
            mask (bool tensor of size equal to the no. of the output units of the networks): tensor indicating which actions are available

        Returns:
            tensor of size [1,1]: The index of the action chosen by the agent
        """
        if not torch.any(mask):
            return None
        
        A_p  = None
        legal = [i for i in range(len(mask)) if mask[i]]
        explore = np.random.random()

        # linear decay of exploration
        eps = max(self.eps_end, self.eps_start - (self.steps / self.eps_decay) * (self.eps_start - self.eps_end))

        # exponential decay of exploration
        # eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps/self.eps_decay)
        
        # explore
        if explore < eps:
            A_p = torch.tensor([[np.random.choice(legal)]],device=device, dtype=torch.long)
     
        # exploit: choose the move with the largest Q estimate from the available ones (i.e. filtered with mask)
        else:
            with torch.no_grad():
               action = torch.argmax(self.behavior_net(state)[mask])  
               j = 0
               for i in range(len(mask)):
                   if mask[i]:
                       if j == action:
                           A_p = torch.tensor([[i]])
                           break
                       else:
                           j += 1
        
        # increment number of actions taken
        self.steps += 1

        return A_p.to(device)
    
    
    def update_agent(self):
        """
        The function that updates the networks according to DDQN algorithm
        """
        # don't update if the experience replay buffer is smaller than the update batch size
        if len(self.mem) < self.batch_size:
            return
        
        # randomly choose the update batch from the experience replay buffer
        transitions = self.mem.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        # mask indicating which next states are not terminal
        non_final_mask = torch.tensor(tuple(map(lambda s_p: s_p is not None, batch.S_p)), device=device, dtype=torch.bool)

        non_final_next_states = torch.cat([s_p for s_p in batch.S_p if s_p is not None])
        s_batch = torch.cat(batch.S)
        a_batch = torch.cat(batch.A)
        r_batch = torch.cat(batch.R)

        # the values Q(s,a) for each of the transition tuples, where Q is the policy network
        act_space = self.behavior_net(s_batch).gather(1, a_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)

        # the values max_(a') over Q'(s',*) where Q' is the target network and s' are the next states which are not terminal
        with torch.no_grad():
            next_state_values[non_final_mask] = torch.max(self.target_net(non_final_next_states),1)[0]

        # the new estimates according to the DDQN update
        pred = self.alpha * ((self.gamma * next_state_values) + r_batch)

        # update the policy network
        criterion = nn.SmoothL1Loss()
        loss = criterion(act_space, pred.unsqueeze(1))    
        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decrease optimizer's learning rate once the loss stagnates
        # self.scheduler.step(loss)
        if self.steps % 50000 == 0:
            self.scheduler.step()


        if self.softUpdates:
            # soft updates: every update, recompute the target network, according to: 
            # new_target_net_weights = tau * policy_network_weights + (1- tau) * current_target_net_weights
            target_net_state_dict = self.target_net.state_dict()
            behavior_net_state_dict = self.behavior_net.state_dict()
            for key in behavior_net_state_dict:
                target_net_state_dict[key] = behavior_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
        else:
            # hard updates: every tau updates copy all parameters from policy network to target network
            if self.steps % self.tau == 0:
                print("TRAIN: Cloning the behavior network into target network")
                self.target_net.load_state_dict(self.behavior_net.state_dict())



    def step_agent(self, A, S_p, R, mask, terminal, turn):
        """Function which, when called, chooses the move, updates the memory buffer and updates the networks
        Args:
            A (tensor): the last action taken, which now will be added to the transition
            S_p (_type_): the state resulted by the action A taken by the agent (might include the opponent's action as well)
            R (long): the reward for taking action A
            mask (Bool tensor): mask for the actions available in the state S_p
            terminal (Bool): indicator whether the game has terminated
            turn : indicator whose player's turn it is. Used primarly when the agent takes actions for both players

        Returns:
            tensor of size [1,1]: The action taken by the agent in state S_p.
            If the game is terminated (i.e. terminal==True), None is returned
        """
        A_p = None
        if not terminal:
            A_p = self.move(S_p, mask)
        
        # perform action for player 0
        if turn == -1: 
            if self.curr_state_p0 is not None:
                self.mem.push(torch.unsqueeze(self.curr_state_p0,0), 
                                A,
                                S_p if S_p is None else torch.unsqueeze(S_p,0), 
                                torch.tensor([R],device=device))
            self.curr_state_p0 = S_p
        # perform action for player 1
        else: 
            if self.curr_state_p1 is not None:
                self.mem.push(torch.unsqueeze(self.curr_state_p1,0), 
                                A,
                                S_p if S_p is None else torch.unsqueeze(S_p,0), 
                                torch.tensor([R],device=device))
            self.curr_state_p1 = S_p

        self.update_agent()

        return A_p
    
class DQNPlayer():
    """
    Class implementing a player which (optionally starts the game with a random move
    and afterwards) plays according to the greedy policy given a trained network.

    Used primarly for evaluation of the networks
    """
    def __init__(self, in_obs, out_actions, weights):
        self.network = deepQNet(in_obs, out_actions)
        self.network.load_state_dict(weights)
        self.network.eval()
        

    def step_agent(self, A, S_p, R, mask, terminal):
        if terminal:
            return None
        else:
            # if all actions are available, choose the action uniformly at random
            if torch.all(mask):
                ran = mask.size()[0]
                return torch.randint(low=0, high=ran, size=(1,1))
            
            # otherwise choose the action according to the policy
            with torch.no_grad():
               action = torch.argmax(self.network(S_p)[mask])       
               j = 0
               for i in range(len(mask)):
                   if mask[i]:
                       if j == action:
                           A_p = torch.tensor([[i]])
                           break
                       else:
                           j += 1
            return A_p.to(device)
