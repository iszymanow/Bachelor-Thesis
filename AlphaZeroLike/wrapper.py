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
from alphaAgent import AlphaAgent
import copy

#pc:
sys.path.append('/home/igor/Bachelor-Thesis/checkers/')
import env




def trainDistributed(numWorkers, numGames, eval_frequency):

    
    if __name__ == '__main__':
        game_inst = env.Env()
        agent = AlphaAgent(game_inst, 6,512,1)
        mp.set_start_method('spawn')
        agent.net.share_memory()
        agent.losses = mp.Manager().list(agent.losses)
        thinkingTime = 100
        temp = 1
        c_puct = 1
        dir_noise = 0.3
        eps = 0.25
        minibatch = 32
        val = mp.Value('i', 0)

        torch.save(agent.net, "challenger.pt")

        

        barrier = mp.Barrier(numWorkers)
        results = mp.Manager().dict()
        evalGames = 100
        evalThinkingTime = 50


        work = [mp.Process(target=wrapper, args=("challenger.pt",agent,numWorkers, eval_frequency//numWorkers, thinkingTime, temp, c_puct, dir_noise, eps,val,barrier,minibatch, evalGames//numWorkers, evalThinkingTime, results, i)) for i in range(numWorkers)]

        for p in work:
            p.start()

        # i = 0
        # while True:
        #     with val.get_lock():
                
        #         if val.value == (i+1) * eval_frequency:
                    

        #             # results = mp.Manager().dict()
        #             # evalGames = 100
        #             # evalThinkingTime = 8
        #             # chal = torch.load('challenger.pt')
        #             # eval_work = [mp.Process(target=agent.evaluate, args=(chal, evalGames/numWorkers, evalThinkingTime, results, j,)) for j in range(numWorkers)]
        #             # for p in eval_work:
        #             #     p.start()

        #             # for p in eval_work:
        #             #     p.join()
        #             barrier.wait()
        #             agentWins = 0
        #             draws = 0
        #             challengerWins = 0
        #             for j in range(len(work)):
        #                 agentWins += results[j][0]
        #                 draws += results[j][1]
        #                 challengerWins += results[j][2]
                    
        #             print("agent's wins: " + str(agentWins) + ", draws: " + str(draws) + ", challenger's wins: " + str(challengerWins))

        #             if agentWins/(agentWins + draws + challengerWins) > 0.55:
        #                 print("new challenger!")
        #                 torch.save(agent.net, "challenger.pt")
        #             else:
        #                 agent.net = torch.load("challenger.pt")

        #             i += 1
        #     if (i+1) * eval_frequency > numGames:
        #         break
        for p in work:
            p.join()

def wrapper(selfplay_agent_path, train_agent, iterations, local_eval_frequency, thinking_time, temp,c_puct, dir_noise, eps, lock, barrier, minibatch, evalGames, evalThinkingTime, results, idx):
        for i in range(iterations):
            # obtain a copy for selfplay
            selfplay_agent = AlphaAgent(train_agent.env,6,512,1)
            selfplay_agent.net = copy.deepcopy(train_agent.net)
            selfplay_agent = copy.deepcopy(train_agent)
            selfplay_agent.net = torch.load(selfplay_agent_path)
            # move the transition tuples to the training agent
            train_agent.mem.memory = selfplay_agent.mem.memory
            selfplay_agent.selfplay(local_eval_frequency, thinking_time, temp, c_puct, dir_noise, eps, lock)
            train_agent.train(1000, minibatch)

            barrier.wait()
            print(np.mean(train_agent.losses))

            train_agent.evaluate(selfplay_agent.net, evalGames, evalThinkingTime, results, idx)
            barrier.wait()
            if idx == 0:
                agentWins = 0
                draws = 0
                challengerWins = 0
                for j in range(len(results)):
                    agentWins += results[j][0]
                    draws += results[j][1]
                    challengerWins += results[j][2]

                print("agent's wins: " + str(agentWins) + ", draws: " + str(draws) + ", challenger's wins: " + str(challengerWins))
                if agentWins/(agentWins + draws + challengerWins) > 0.55:
                    print("new challenger!")
                    torch.save(train_agent.net, selfplay_agent_path)
                else:
                    train_agent.net = selfplay_agent.net
            barrier.wait()
            
            
def main():
    trainDistributed(20,10000, 1000)

main()
