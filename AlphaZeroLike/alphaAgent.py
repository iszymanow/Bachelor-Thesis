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
from neuralNets import CheckersNN, TicTacToeNN
import sys
from tqdm import tqdm
import os
import copy

#pc:
sys.path.append('/home/igor/Bachelor-Thesis/checkers/')
import env
import envTicTacToe

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device='cpu'
dataSample = namedtuple('dataSample', ('s_t', 'pi_t', 'z_t',))


class CustomLoss(nn.Module):
    def __init__(self, model, reg):
        super(CustomLoss, self).__init__()
        self.model = model
        self.reg = reg
    
    def forward(self, output_v, output_p, target_v, target_p):
        mse = torch.mean((output_v - target_v) ** 2)
        ce = -torch.mean(torch.sum(target_p * torch.log(output_p),1))
        l2 = 0
        for param in self.model.parameters():
            l2 += torch.sum(param.data**2)

        loss = mse + ce + self.reg * l2

        return loss
    
class ReplayMemory(object):
    """ 
    The implementation of Experience Replay buffer which stores the update transition. The default size is 10000 transitions
    """
    def __init__(self, capacity=8000):
        self.memory = []
        self.game_length = deque()
        self.game_ctr = 0
        self.cap = capacity

    def push(self, data):
        self.memory.append(data)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    def extend(self, list):
        self.memory += list
        self.game_length.append(len(list))
        self.game_ctr += 1
        if self.game_ctr >= self.cap:
            toPop = self.game_length.popleft()
            self.memory = self.memory[toPop:]

        # print(sys.getsizeof(self.memory))
        

class AlphaAgent():
    def __init__(self, env, net) -> None:
        self.env = env
        self.net = net.to(device)
        self.mem = ReplayMemory()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-2, amsgrad=False)
        self.criterion = CustomLoss(model=self.net, reg=1e-4)
        self.losses = []


    def move(self, probs):
        m = distr.Categorical(probs)
        return m.sample()

    def selfplay(self, numGames, thinkingTime, c_init, c_base, dir_noise, eps):
        self.net.eval()
        mcts = MCTS(self.env, self.net, thinkingTime, c_init, c_base, dir_noise, eps)
        for i in tqdm(range(numGames)):
            i_states, i_probs, i_result = [], [], []
            self.env.reset()            

            # the main episode loop
            while True:
                S_prime, mask, isTerminated = self.env.get_obs()
                # self.env.render()
                # print(isTerminated)
                if isTerminated:
                    
                    R, R_2 = self.env.step_env(None)
                    i_result = [torch.tensor([R], dtype=torch.float) if x == -1 else torch.tensor([R_2], dtype=torch.float) for x in i_result]
                    i_buffer = list(zip(i_states, i_probs, i_result))
                       
                    self.mem.extend(i_buffer)
                   
                    break

                else:
                    env_copy = self.env.save()

                    probs = mcts.getDistribution(env_copy)
                    s = self.env.encodeObs().unsqueeze(0)
                    i_states.append(s)
                    i_probs.append(probs.unsqueeze(0))
                    i_result.append(self.env.turn)

                    A = self.move(probs)
                    # print(A)

                    self.env.step_env(A)
            
        


        print("p0 wins: " + str(self.env.p0_wins) + 
               " draws: " + str(self.env.draws) +
               " p1 wins: " + str(self.env.p1_wins))
        


    def train(self, iters, minibatch):
        if self.mem.__len__() < minibatch:
            return
        else:
            self.net.train()
            for i in tqdm(range(iters)):
                data = self.mem.sample(minibatch)
                
                batch = dataSample(*zip(*data))

                s_batch = torch.cat(batch.s_t).to(device)
                target_pi = torch.cat(batch.pi_t).to(device)
                target_v = torch.stack(batch.z_t).to(device)
                output_pi, output_v = self.net(s_batch)


                self.optimizer.zero_grad()

                loss = self.criterion(output_v, output_pi, target_v, target_pi)
                self.losses.append(loss.item())

                loss.backward()

                self.optimizer.step()


    def evaluate(self, challengerNet, numGames, thinkingTime, results, idx):
        game_inst = copy.copy(self.env)
        self.net.eval()
        challengerNet.eval()
        p0 = MCTS(game_inst, self.net, thinkingTime, 1.25 ,19625, 0.3, 0.25)
        p1 = MCTS(game_inst, challengerNet, thinkingTime, 1.25 ,19625, 0.3, 0.25)
        # iterator = iter(self.net.parameters())
        # for param in challengerNet.parameters():
        #     print((param.data == next(iterator)).all())
        agentWins = 0
        challengerWins = 0
        draws = 0

        for i in tqdm(range(int(numGames))):

            game_inst.reset()
            start = np.random.choice([-1,1])
            # if start == -1:
                # print('agent starts!')

            while True:
                    S_prime, mask, isTerminated = game_inst.get_obs()
                    # print(isTerminated)
                    if isTerminated:
                        game_inst.render()

                        R, R_2 = game_inst.step_env(None)
                        if start == -1:
                            if R == 1:
                                agentWins += 1
                                # print('agent won')
                            elif R == -1:
                                challengerWins += 1
                                # print('challenger won')
                            else:
                                draws += 1
                        else:
                            if R == 1:
                                # print('challenger won')
                                challengerWins += 1
                            elif R == -1:
                                # print('agent won')

                                agentWins += 1
                            else:
                                draws += 1
                        break

                    else:
                        params = game_inst.save()
                        if game_inst.turn == -1:
                            if start == -1:
                                probs = p0.getDistribution(params, greedy=True)
                                # print(probs)
                            else:
                                probs = p1.getDistribution(params, greedy=True)
                        else:
                            if start == -1:
                                probs = p1.getDistribution(params, greedy=True)
                            else:
                                probs = p0.getDistribution(params, greedy=True)
                                # print(probs)
                        A = self.move(probs)
                        # print(A)
                        # game_inst.render()
                        game_inst.step_env(A)

        results[idx] = [agentWins, draws, challengerWins]
        print("agent's wins: " + str(agentWins) + ", draws: " + str(draws) + ", challenger's wins: " + str(challengerWins))

    def mainLoop(self):
        challenger = copy.deepcopy(self.net)
        for i in tqdm(range(30)):
            state_dict = self.net.state_dict()
            challenger.load_state_dict(state_dict)
            self.selfplay(50,50, 1.25, 19265, 0.3, 0.25)
            
            self.train(10,32)
            print(np.mean(self.losses))

            results = {}
            self.evaluate(challenger, 20, 50, results,0)
            agentWins = results[0][0]
            draws = results[0][1]
            challengerWins = results[0][2]
            if (agentWins + draws)/(agentWins + challengerWins + draws) > 0.55:
                print("new challenger!" + str(i))
            else:
                self.net.load_state_dict(state_dict)
        
        torch.save(self.net, "final_mainLoop.pt")


    def trainDistributed(self, numWorkers, numGames, eval_frequency):

    
        if __name__ == '__main__':
            mp.set_start_method('spawn')
            self.net.share_memory()
            self.losses = mp.Manager().list(self.losses)
            thinkingTime = 8
            c_init = 1.25
            c_base = 19625
            dir_noise = 0.3
            eps = 0.25
            minibatch = 64
            val = mp.Value('i', 0)

            torch.save(self.net, "challenger.pt")

            barrier = mp.Barrier(numWorkers)
            selfplay_training_work = [mp.Process(target=self.wrapper, args=(numWorkers, eval_frequency//numWorkers, thinkingTime, c_init, c_base, dir_noise, eps,val,barrier,minibatch)) for i in range(numWorkers)]

            for p in selfplay_training_work:
                p.start()


            i = 0
            while True:
                with val.get_lock():
                   
                    if val.value == (i+1) * eval_frequency:
                        

                        results = mp.Manager().dict()
                        evalGames = 10
                        evalThinkingTime = 4
                        chal = torch.load('challenger.pt')
                        eval_work = [mp.Process(target=self.evaluate, args=(chal, evalGames/numWorkers, evalThinkingTime, results, j,)) for j in range(numWorkers)]
                        for p in eval_work:
                            p.start()

                        for p in eval_work:
                            p.join()

                        agentWins = 0
                        draws = 0
                        challengerWins = 0
                        for j in range(len(eval_work)):
                            agentWins += results[j][0]
                            draws += results[j][1]
                            challengerWins += results[j][2]
                        
                        print("agent's wins: " + str(agentWins) + ", draws: " + str(draws) + ", challenger's wins: " + str(challengerWins))

                        if agentWins/(agentWins + draws + challengerWins) > 0.55:
                            torch.save(self.net, "challenger.pt")
                        else:
                            self.net = torch.load("challenger.pt")

                        i += 1
                if (i+1) * eval_frequency > numGames:
                    break

            for p in selfplay_training_work:
                p.join()


    def wrapper(self, iterations, local_eval_frequency, thinking_time, temp,c_puct, dir_noise, eps, lock, barrier, minibatch):
            for i in range(iterations):
                self.selfplay(local_eval_frequency, thinking_time, temp, c_puct, dir_noise, eps, lock)
                barrier.wait()
                self.train(1000, minibatch)
                barrier.wait()
                # print(np.mean(self.losses))



            

def main():
    # game_inst = env.Env()
    # agent = AlphaAgent(game_inst, 6,512,2)
    # agent.mainLoop()
    # agent.trainDistributed(10, 100, 10)

    # torch.save(agent.net, "final_version.pt")
    # net = AlphaAgent(game_inst,6,512,1)
    # net.net = torch.load("challenger.pt")
    # agent.evaluate(net.net,1,10,dict(), 0)

    # game_inst = envTicTacToe.Env()
    # ticNet = TicTacToeNN(1)
    # ticAgent = AlphaAgent(game_inst, ticNet)
    # ticAgent.selfplay(5, 200, 1.25, 19265, 0.3, 0.25)
    # ticAgent.mainLoop()
    # ticAgent.evaluate(ticNet, 10, 100,dict(),0)
    check_env = env.Env()
    checkNet = CheckersNN(1)
    checkAgent = AlphaAgent(check_env, checkNet)
    checkAgent.mainLoop()
    # testPos = np.array(
        # [0,  0,  0,  0,  0,  0,  0,  0,
        #  0,  0,  0, -2,  0,  0,  0,  0,
        #  0,  0,  0,  0,  0,  0,  0,  0,
        #  0,  0,  0,  1,  0,  0,  0,  0,
        #  1,  0,  1,  0,  1,  0,  0,  0,
        #  0,  0,  0,  0,  0,  0,  0,  0, 
        #  0,  0,  0,  0,  0,  0,  0,  0,
        #  0,  0,  0,  0,  0,  1,  0,  0])
    # print(testPos.shape)
    # print(check_env.state.shape)
    # check_env.turn = 1
    # game_inst.state = testPos
    # check_env.state = testPos
    # check_env.positions[tuple(testPos.tolist())] = 0

    # mcts = MCTS(check_env, checkNet, 50, 1.25, 19265, 0.25, 0)
    # probs = mcts.getDistribution(check_env, greedy=True)
    # check_env.step_env(mcts.move(probs))
    # check_env.render()

# main()