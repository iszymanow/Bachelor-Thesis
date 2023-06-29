import mcts
import sys
from tqdm import tqdm
import numpy as np
import torch

import env
import envTicTacToe
from mcts import MCTS
import alphaAgent
import neuralNets
import player0


def testRandom(env, testNet, thinkingTime, numGames):
    game_inst = env
    p0 = mcts.MCTS(game_inst, testNet, thinkingTime, 1.25, 19265, 0.3, 0.25)
    p1 = player0.RandomPlayer()

    agentWins = 0
    randomWins = 0
    draws = 0

    for i in tqdm(range(numGames)):

        game_inst.reset()
        start = np.random.choice([-1,1])
        # print(start)

        while True:
                S_prime, mask, isTerminated = game_inst.get_obs()
                mask = mask.reshape(-1)
                # print(isTerminated)
                # game_inst.render()

                if isTerminated:
                    R, R_2 = game_inst.step_env(None)
                    if start == -1:
                        if R == 1:
                            agentWins += 1
                        elif R == -1:
                            randomWins += 1
                        else:
                            draws += 1
                    else:
                        if R == 1:
                            randomWins += 1
                        elif R == -1:
                            agentWins += 1
                        else:
                            draws += 1
                    break

                else:

                    params = game_inst.save()
                    if game_inst.turn == -1:
                        if start == -1:
                            probs = p0.getDistribution(params, greedy=True)
                            A = p0.move(probs)
                        else:
                            A = p1.step_agent(mask)
                    else:
                        if start == -1:
                             A = p1.step_agent(mask)
                        else:
                            probs = p0.getDistribution(params, greedy=True)
                            A = p0.move(probs)


                    game_inst.step_env(A)

    print("agent's wins: " + str(agentWins) + ", draws: " + str(draws) + ", random's wins: " + str(randomWins))

def evaluate(env, p0_net, p1_net, numGames, thinkingTime, results, idx):
        game_inst = env
        p0_net.eval()
        p1_net.eval()
        p0 = MCTS(game_inst, p0_net, thinkingTime, 1.25 ,19625, 0.3, 0)
        p1 = MCTS(game_inst, p1_net, thinkingTime, 1.25 ,19625, 0.3, 0)
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
                                A = p0.move(probs)
                            else:
                                probs = p1.getDistribution(params, greedy=True)
                                A = p1.move(probs)
                        else:
                            if start == -1:
                                probs = p1.getDistribution(params, greedy=True)
                                A = p1.move(probs)
                            else:
                                probs = p0.getDistribution(params, greedy=True)
                                A = p0.move(probs)
                        # print(A)
                        # game_inst.render()
                        game_inst.step_env(A)

        results[idx] = [agentWins, draws, challengerWins]
        print("p0's wins: " + str(agentWins) + ", draws: " + str(draws) + ", p1's wins: " + str(challengerWins))

def main():
#     net = torch.load('final_mainLoop.pt')

    net2 = torch.load('checkpoints_tictactoe/final_version.pt')

    # net2 = torch.load('checkpoints_checkers/checkpoint_2.pt')
    # net2 = torch.load('checkpoints_checkersfinal_version.pt')

#     # iterator = iter(net2.parameters())
#     # for param in net.parameters():
   
#     #     print(torch.eq(param.data, next(iterator).data).all())
    ticNet = neuralNets.TicTacToeNN(1)
    ticEnv = envTicTacToe.Env()
#     # ticAgent = alphaAgent.AlphaAgent(ticEnv, ticNet)
#     testRandom(ticEnv, net, 10, 100)
#     # evaluate(ticEnv, net, net2, 100, 25, dict(),0)
    checkEnv = env.Env()
    checkNetInit = neuralNets.CheckersNN(1)

    testRandom(ticEnv, net2, 10, 100)

main()