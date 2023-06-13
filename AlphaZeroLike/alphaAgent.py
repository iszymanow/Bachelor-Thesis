import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distr
import torch.multiprocessing as mp
import random
from collections import namedtuple, deque
import numpy as np

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
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3, amsgrad=True)
        self.losses = []


    def move(self, probs):
        # print(probs)
        m = distr.Categorical(probs)
        return m.sample()

    def selfplay(self, numGames, thinkingTime, temp, c_puct, dir_noise, eps):
        self.net.eval()
        mcts = MCTS(self.env, self.net, thinkingTime, temp, c_puct, dir_noise, eps)

        for i in tqdm(range(numGames)):
            i_states, i_probs, i_result = [], [], []
            self.env.reset()
            numMoves = 0

            # the main episode loop
            while True:
                    S_prime, mask, isTerminated = self.env.get_obs()
                    # self.env.render(orient=self.env.turn)
                    # print(isTerminated)
                    if isTerminated:
                        R, R_2 = self.env.step_env(None)
                        i_result = [R if x == -1 else R_2 for x in i_result]
                        i_buffer = list(zip(i_states, i_probs, i_result))
                        self.mem.extend(i_buffer)
                        self.env.render(orient=self.env.turn)

                        break

                    else:
                        numMoves += 0.5
                        if numMoves > 30:
                            mcts.temperature = 1/numMoves
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


    def train(self, minibatch):
        if self.mem.__len__() < minibatch:
            return
        else:
            self.net.train()
            data = self.mem.sample()
            
            batch = dataSample(*zip(*data))

            s_batch = torch.cat(batch.s_t)
            target_pi = torch.cat(batch.pi_t)
            target_v = torch.cat(batch.z_t)

            output_pi, output_v = self.net(s_batch)

            criterion = CustomLoss(model=self.net, reg=1)
            loss = criterion(output_v, output_pi, target_v, target_pi)
            self.losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()


    def evaluate(self, challengerNet, numGames, thinkingTime):
        game_inst = self.env
        self.net.eval()
        challengerNet.eval()
        p0 = MCTS(game_inst, self.net, thinkingTime, 0.001 ,1, 0, 0)
        p1 = MCTS(game_inst, challengerNet, thinkingTime, 0.001 ,1, 0, 0)
        whoStarts = []
        agentWins = 0
        challengerWins = 0
        draws = 0

        for i in tqdm(range(numGames)):
            game_inst.reset()
            start = np.random.choice([-1,1])
            whoStarts.append(start)

            while True:
                    S_prime, mask, isTerminated = self.env.get_obs()
                    # self.env.render(orient=self.env.turn)
                    # print(isTerminated)
                    if isTerminated:
                        R, R_2 = self.env.step_env(None)
                        if whoStarts[i] == -1:
                            if R == 1:
                                agentWins += 1
                            elif R == -1:
                                challengerWins += 1
                            else:
                                draws += 1
                        else:
                            if R == 1:
                                challengerWins += 1
                            elif R == -1:
                                agentWins += 1
                            else:
                                draws += 1
                        break

                    else:
                        params = self.env.save()
                        if game_inst.turn == -1:
                            if whoStarts == -1:
                                probs = p0.getDistribution(params, greedy=True)
                            else:
                                probs = p1.getDistribution(params, greedy=True)
                        else:
                            if whoStarts == -1:
                                probs = p1.getDistribution(params, greedy=True)
                            else:
                                probs = p0.getDistribution(params, greedy=True)

                        A = self.move(probs)

                        self.env.step_env(A)

        print("agent's wins: " + str(agentWins) + ", draws: " + str(draws) + ", challenger's wins: " + str(challengerWins))

    def mainLoop(self):
        for i in tqdm(range(10)):
            state_dict = self.net.state_dict().copy()
            self.selfplay(1028,100, 1, 1, 0.3, 0.25)
            for j in tqdm(range(100)):
                self.train(32)
            challenger = CheckersNN(6,512,1)
            challenger.load_state_dict(state_dict)
            self.evaluate(challenger, 400, 400)


def main():
    game_inst = env.Env()
    agent = AlphaAgent(game_inst, 6,512,1)
    agent.mainLoop()



main()