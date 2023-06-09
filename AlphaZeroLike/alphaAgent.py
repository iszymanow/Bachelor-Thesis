import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distr
import random
from collections import namedtuple, deque

from mcts import MCTS
from neuralNets import CheckersNN
import sys
from tqdm import tqdm

#pc:
sys.path.append('/home/igor/Bachelor-Thesis/checkers/')
import env

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device='cpu'
dataSample = namedtuple('dataSample', ('s_t', 'pi_t', 'z_t',))


class CustomLoss(nn.Module):
    def __init__(self, model, reg):
        super(CustomLoss, self).__init__()
        self.model = model
        self.reg = reg
    
    def forward(self, output_v, output_p, target_v, target_p):
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        l2 = 0
        for param in self.model.parameters:
            l2 += torch.sum(param.data**2)

        loss = mse(output_v, target_v) + ce(output_p, target_p) + self.reg * l2

        return loss
    
class ReplayMemory(object):
    """ 
    The implementation of Experience Replay buffer which stores the update transition. The default size is 10000 transitions
    """
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, data):
        self.memory.append(data)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    def extend(self, list):
        self.memory.extend(list)

class AlphaAgent():
    def __init__(self, env, numObs, numActions, numResBlocks) -> None:
        self.env = env
        self.net = CheckersNN(numObs, numActions, numResBlocks)
        self.mem = ReplayMemory()

    def move(self, probs):
        m = distr.Categorical(probs)
        return m.sample()

    def selfplay(self, numGames, thinkingTime, temp, c_puct, dir_noise, eps):
        self.net.eval()
        mcts = MCTS(self.env, self.net, thinkingTime, temp, c_puct, dir_noise, eps)

        for i in tqdm(range(numGames)):
            i_states, i_probs, i_result = [], [], []
            self.env.reset()

            # the main episode loop
            while True:
                    S_prime, mask, isTerminated = self.env.get_obs()
                    self.env.render(orient=self.env.turn)
                    # print(isTerminated)
                    if isTerminated:
                        R, R_2 = self.env.step_env(None)
                        i_result[i_result == -1] = R
                        i_result[i_result == 1] = R_2
                        i_buffer = list(zip(i_states, i_probs, i_result))
                        self.mem.extend(i_buffer)
                        break

                    else:
                        params = self.env.save()
                        s = str(self.env)
                        probs = mcts.getDistribution(params)

                        i_states.append(s)
                        i_probs.append(probs)
                        i_result.append(self.env.turn)

                        A = self.move(probs)
                        # print(A)

                        self.env.step_env(A)

        print("p0 wins: " + str(self.env.p0_wins) + 
             " p1 wins: " + str(self.env.p1_wins) +
               " draws: " + str(self.env.draws))




def main():
    game_inst = env.Env()
    agent = AlphaAgent(game_inst, 6,512,1)
    agent.selfplay(1,10, 1, 1, 0.3, 0.25)



main()