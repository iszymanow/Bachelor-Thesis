import numpy as np
import torch
import torch.distributions as distr
import copy


class MCTS():
    
    def __init__(self, env, network, sims, c_init, c_base, dir_noise, eps) -> None:
        self.env = env
        self.net = network
        self.device = next(self.net.parameters()).device
        # for param in network.parameters():
        #     print(param.data)
        self.sims = sims
        self.c_init = c_init
        self.c_base = c_base
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
        # self.env.render()

        self.env = rollback_env.save()
        s0 = str(self.env)

        #TODO solve issue with envs
        for i in range(self.sims):
            v = self.rollout()
            # print("after:")
            # self.env.render()
            self.env = copy.deepcopy(rollback_env)
            # print("returned:", v)
        # self.env.render()
        # print("P:",self.P[s0][self.masks[s0]])
        # print("N:",self.N[s0][self.masks[s0]])
        # print("Q:",self.Q[s0][self.masks[s0]])
        # print("W:",self.W[s0][self.masks[s0]])
        # print("v: ", v)
        # print('\n\n')
        assert(s0 == str(self.env))

        
        policy = self.N[s0] / np.sum(self.N[s0])
        # print("dream policy:", policy[self.masks[s0]])
        
        # print(policy)
        if greedy:
            # choose the most frequently explored move (with random tie-breaking)
            argmax = np.random.choice(np.flatnonzero(policy == policy.max()))
            policy = np.zeros_like(policy)
            policy[argmax] = 1
        
        # print(policy)
        return torch.tensor(policy)


    def rollout(self):
        s = str(self.env)
        # print(s)
        # print(self.env.turn)


        # the state has been encountered for the first time
        if self.masks.get(s) is None:
            mask = self.env.get_mask(self.env.turn).reshape(-1)
            isDone = self.env.isTerminated(mask)

            self.isTerminal[s] = isDone
            self.masks[s] = mask
            # print(mask.any())
        
        # the state is terminal (the choice of num 3 defined by design of the environment)
        if self.isTerminal[s] != 3:
            return self.isTerminal[s]
        else:
            # a leaf node, hasn't been explored before
            if self.P.get(s) is None:
                with torch.no_grad():
     
                    encoded = self.env.encodeObs().to(self.device)
                    p, v = self.net(encoded.unsqueeze(0))
                self.P[s] = p.cpu().numpy().reshape(-1)

                # print("new position", v)
    
                # add the noise to the distribution to regulate the exploration
                noise = np.random.dirichlet([self.dir_noise] * len(self.P[s][self.masks[s]]))
                # print(noise)
                # print("before adding noise:",self.P[s][self.masks[s]])
                self.P[s][self.masks[s]] = ((1 - self.eps) * self.P[s][self.masks[s]] + self.eps * noise)
                self.P[s] *= self.masks[s]
                # print("after adding noise",self.P[s][self.masks[s]])
            

                self.P[s] /= np.sum(self.P[s])


                
                

                # print("sum:",np.sum(self.P[s]))
                self.N[s] = np.zeros_like(self.masks[s], dtype=float)
                self.W[s] = np.zeros_like(self.masks[s], dtype=float)
                self.Q[s] = np.zeros_like(self.masks[s], dtype=float)

            # not a leaf node, choose the best action w.r.t. PUCT algorithm
            else:
                c_puct = np.log(np.sum(self.N[s]) + self.c_base + 1)/self.c_base + self.c_init

                U = c_puct * self.P[s] * np.sqrt(np.sum(self.N[s])) / (1 + self.N[s])
                # print("U:",U)
                # print("Q:",self.Q[s])
                action_values = U + self.Q[s]
                action_values[~self.masks[s]] = -np.inf
                # print("c_puct:",c_puct)

               
                # print("A_V:",action_values)

                a = np.argmax(action_values)
                # print(a)
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
                self.N[s][a] += 1
                self.W[s][a] += v
                self.Q[s][a] = self.W[s][a] / self.N[s][a]

            # print(v)
            return v
