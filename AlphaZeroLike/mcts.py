import numpy as np
import torch
import torch.distributions as distr
from tqdm import tqdm
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


class MCTS():
    
    def __init__(self, env, network, sims, temperature, c_puct, dir_noise, eps) -> None:
        self.env = env
        self.net = network
        # for param in network.parameters():
        #     print(param.data)
        self.sims = sims
        self.temperature = temperature
        self.c_puct = c_puct
        self.dir_noise = dir_noise
        self.eps = eps
        self.N = {}
        self.W = {}
        self.Q = {}
        self.P = {}

        self.isTerminal = {}
        self.masks = {}
    
    def move(self, probs):
        m = distr.Categorical(probs)
        return m.sample()
    
    def getDistribution(self, rollback_env, greedy=False):
        # discard the previous tree
        self.N = {}
        self.W = {}
        self.Q = {}
        self.P = {}

        self.isTerminal = {}
        self.masks = {}
        # print("before:")
        # self.env.render(orient=1)
        s0 = str(rollback_env)
        for i in range(self.sims):
            v = self.rollout()
            # print("after:")
            # self.env.render(orient=1)
            self.env = copy.deepcopy(rollback_env)
            # print("returned:", v)
       
        assert(s0 == str(self.env))

        policy = (self.N[s0] ** (1/self.temperature)) / (np.sum(self.N[s0] ** 1/self.temperature))
        

        if greedy:
            argmax = np.argmax(policy)
            policy = np.zeros_like(policy)
            policy[argmax] = 1
        
        # print(policy)
        return torch.tensor(policy)


    def rollout(self):
        s = str(self.env)


        # the state has been encountered for the first time
        if self.isTerminal.get(s) is None:
            mask = self.env.get_mask(self.env.turn).reshape(-1)
            isDone = self.env.isTerminated(mask)

            self.isTerminal[s] = isDone
            self.masks[s] = mask
        
        # the state is terminal (the choice of num 3 defined by design of the environment)
        if self.isTerminal[s] != 3:
            return self.isTerminal[s]
        else:
            # a leaf node, hasn't been explored before
            if self.P.get(s) is None:
                with torch.no_grad():
                    encoded = self.env.encodeObs().to(device)
                    p, v = self.net(encoded.unsqueeze(0))
                self.P[s] = p.cpu().numpy().reshape(-1)
                # print("new position", v)
    
                # add the noise to the distribution to regulate the exploration
                noise = np.random.dirichlet([self.dir_noise], self.P[s].shape).reshape(-1)
                # print("noise:\n",noise)

                self.P[s] = ((1 - self.eps) * self.P[s] + self.eps * noise)
                self.P[s] *= self.masks[s]
                self.P[s] /= np.sum(self.P[s])
                # print("sum:",torch.sum(self.P[s]))
                self.N[s] = np.zeros_like(mask, dtype=float)
                self.W[s] = np.zeros_like(mask, dtype=float)
                self.Q[s] = np.zeros_like(mask, dtype=float)

            # not a leaf node, choose the best action w.r.t. PUCT algorithm
            else:
                U = self.c_puct * self.P[s] * np.sqrt(np.sum(self.N[s])) / (1 + self.N[s])
                action_values = U + self.Q[s]
                action_values[~self.masks[s]] = -np.inf
                # print(action_values)

                a = np.argmax(action_values)
                turn = self.env.turn
                # take the action a and update the state
                self.env.step_env(a)
                # recursive call on the updated state
                # print("rolling out")
                if self.env.turn != turn:
                    # print("as opponent")
                    # the other player is on the move, we need to negate their v value 
                    v = -self.rollout()
                else:
                    v = self.rollout()
                    # print("as same player")

                self.N[s][a] += 1
                self.W[s][a] += v
                self.Q[s][a] = self.W[s][a] / self.N[s][a]

            return v
