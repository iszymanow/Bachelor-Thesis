import mcts
import sys
from tqdm import tqdm
import numpy as np
import torch

import env
import envTicTacToe
from mcts import MCTS
import neuralNets
import player0
import os


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
        print(start)

        while True:
                S_prime, mask, isTerminated = game_inst.get_obs()
                mask = mask.reshape(-1)
                # print(isTerminated)
                game_inst.render()

                if isTerminated:
                    print('\n\n',start)
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

def evaluate(env, p0_net, p1_net, numGames, thinkingTime):
        game_inst = env
        p0_net.eval()
        p1_net.eval()
        p0 = MCTS(game_inst, p0_net, thinkingTime, 1.25 ,19625, 0.3, 0.25)
        p1 = MCTS(game_inst, p1_net, thinkingTime, 1.25 ,19625, 0.3, 0.25)
        # iterator = iter(self.net.parameters())
        # for param in challengerNet.parameters():
        #     print((param.data == next(iterator)).all())
        agentWins = 0
        challengerWins = 0
        draws = 0

        for i in tqdm(range(int(numGames))):

            game_inst.reset()
            start = np.random.choice([-1,1])
            if start == -1:
                print('p0 starts!')
            else:
                print("p1 starts!")

            while True:
                    S_prime, mask, isTerminated = game_inst.get_obs()
                    # print(isTerminated)
                    game_inst.render()
                    print(game_inst.nonprogress)

                    if isTerminated:
                        print('\n\n',start)

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

        print("p0's wins: " + str(agentWins) + ", draws: " + str(draws) + ", p1's wins: " + str(challengerWins))

def gameAgainstHuman(env, net, thinkingTime):
    computer = MCTS(env, net, thinkingTime, 1.25, 19265, 0.25, 0.3)

    playAgain = True
    while playAgain:
        env.reset()
        playerSign = 0
        while playerSign == 0:
            whoStarts = input("Choose white or black pieces [W/B]: ")
            match whoStarts:
                case 'B': playerSign = -1
                case 'W': playerSign = 1
                case _: print('Invalid input. \n')


        while True:
            S_prime, mask, isTerminated = env.get_obs()
            if isTerminated:
                break

            mask = mask.flatten()
            
            if env.turn == playerSign:
                # os.system('clear')
                env.render(orient=playerSign, manualAid=True)
                validSquare = False
                while not validSquare:
                    move = input('Choose the square number of the piece you want to move: ')
                    try:
                        move = int(move)
                    except ValueError:
                        print('Invalid input.\n')
                        continue
                    if move < 0 or move > 63:
                        print("Invalid square number\n")
                        continue
                    if S_prime[move] * playerSign > 0 and mask[move * 8: (move+1) * 8].any():
                        validSquare = True
                        validAction = False
                        while not validAction:
                            try:
                                action = int(input('Choose the action you want to take: '))
                            except ValueError:
                                print('Invalid input.\n')
                                continue
                            if mask[8 * move + action]:
                                validAction = True
                                action = np.array([8*move + action])
                    else:
                        print("No piece on the square or no possible moves with the piece\n")
        
            else:
                probs = computer.getDistribution(env, True)
                action = computer.move(probs)


            # playerMove(action, orient=game_inst.turn)
            env.step_env(action)
      
        env.render(orient=playerSign)
        again = input('Game finished. Want to play again? [y/n]: ')
        match again:
            case 'y':
                playAgain = True
            case 'n':
                playAgain = False
            case _:
                playAgain = False

def main():
#     net = torch.load('final_mainLoop.pt')

    # net2 = torch.load('checkpoints_tictactoe/final_version.pt')

    eulerNet1 = torch.load('checkpoint_euler_5.pt')
    eulerNet2 = torch.load('checkpoint_0.pt')
    # net2 = torch.load('checkpoints_checkers_32k_30sims/checkpoint_15.pt')

    # iterator = iter(net.parameters())

    # for param in net2.parameters():
    #     print((next(iterator) == param.data).all())

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

    # testRandom(checkEnv, net2, 50, 10)
    # gameAgainstHuman(checkEnv, eulerNet, 200)
    evaluate(checkEnv,eulerNet1, eulerNet2, 1, 400)

main()