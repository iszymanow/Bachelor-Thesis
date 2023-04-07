import numpy as np
from env import Env

class Sarsa_learner:

    def __init__(self, alpha, eps, gamma):
        self.act_space = {}
        self.alpha = alpha #step size
        self.eps = eps #exploration parameter
        self.gamma = gamma #discounting rate

        self.last_state = None
        self.last_action = None
        self.curr_state = None
        self.curr_action = None
        self.last_reward = 0

    def reset(self):
        self.act_space[self.last_state][self.last_action] +=  self.alpha * (self.last_reward - self.act_space[self.last_state][self.last_action])
        self.last_state = None
        self.last_action = None
        self.curr_state = None
        self.curr_action = None
        self.last_reward = 0


    def move(self, legal_moves, state):
        a2s = np.array2string(state)

        if self.act_space.get(a2s) is None:
            self.act_space[a2s] = {}
            for m in legal_moves:
                if self.act_space[a2s].get(m) is None:
                    self.act_space[a2s][m] = 0


        self.curr_state = a2s
        explore = np.random.random() <= self.eps
        if explore:
           index = np.random.randint(len(legal_moves))
           a = legal_moves[index]
        else: #exploit
            a = max(self.act_space[a2s])
        
        self.curr_action = a
        return a

    def reward(self, value):
        change=0
        if self.last_state is None:
            pass
        else:
            change = self.alpha * (self.last_reward + self.gamma*self.act_space[self.curr_state][self.curr_action] - self.act_space[self.last_state][self.last_action])

            self.act_space[self.last_state][self.last_action] += change
        self.last_state = self.curr_state
        self.last_action = self.curr_action
        self.last_reward = value


    def generate_action_space(self):
        return [[[a,b,c],[d,e,f],[g,h,i]] for a in {-1,0,1} for b in {-1,0,1} for c in {-1,0,1} for d in {-1,0,1} for e in {-1,0,1} for f in {-1,0,1} for g in {-1,0,1} for h in {-1,0,1} for i in {-1,0,1}]

    