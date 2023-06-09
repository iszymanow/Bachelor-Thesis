import numpy as np
import torch
import torch.distributions as distr

class MCTS():
    
    def __init__(self, env, network, sims, temperature, c_puct, dir_noise, eps) -> None:
        self.env = env
        self.net = network
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
    

    def getDistribution(self, env_params, greedy=False):
        # discard the previous tree
        self.N = {}
        self.W = {}
        self.Q = {}
        self.P = {}

        self.isTerminal = {}
        self.masks = {}

        for i in range(self.sims):
            self.env.load(env_params)
            self.rollout()

        self.env.load(env_params)
 
        s0 = str(self.env)

        policy = torch.tensor(np.power(self.N[s0], 1/self.temperature) / np.sum(np.power(self.N[s0], 1/self.temperature)))


        if greedy:
            argmax = torch.argmax(policy).clone()
            policy = torch.zeros_like(policy)
            policy[argmax] = 1
        
        return policy


    def rollout(self):
        s = str(self.env)


        # the state has been encountered for the first time
        if self.isTerminal.get(s) is None:
            mask = self.env.get_mask(self.env.turn).flatten()
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
                    encoded = self.env.encodeObs()
                    p, v = self.net(encoded.unsqueeze(0))
                self.P[s] = p * mask
                # add the noise to the distribution to regulate the exploration
                noise = distr.Dirichlet(self.dir_noise *torch.ones_like(self.P[s])).sample()
                # renormalize the distribution, s.t. it adds up to 1
                self.P[s][mask] /= torch.sum(self.P[s][mask])
                self.P[s] = (1 - self.eps) * self.P[s] + self.eps * noise
                self.N[s] = np.zeros_like(mask)
                self.W[s] = np.zeros_like(mask)
                self.Q[s] = np.zeros_like(mask)

            # not a leaf node, choose the best action w.r.t. PUCT algorithm
            else:
                U = self.c_puct * self.P[s] * np.sqrt(np.sum(self.N[s])) / (1 + self.N[s])
                action_values = U + self.Q[s]
                action_values[~self.masks[s]] = -torch.inf

                a = np.argmax(action_values)
                turn = self.env.turn
                # take the action a and update the state
                self.env.step_env(a)
                # recursive call on the updated state
                if self.env.turn != turn:
                    # the other player is on the move, we need to negate their v value 
                    v = -self.rollout()
                else:
                    v = self.rollout()

                self.N[s][a] += 1
                self.W[s][a] += v
                self.Q[s][a] = self.W[s][a] / self.N[s][a]

            return v
