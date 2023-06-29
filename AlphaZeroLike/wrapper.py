import torch
import torch.multiprocessing as mp
import numpy as np
from alphaAgent import AlphaAgent
import neuralNets
import copy

import env
import envTicTacToe




def trainDistributed(net, env, numCPUs, numGPUs, numGames, eval_frequency, evalGames, checkpoints_path):
    if __name__ == '__main__':
        game_inst = env
        agent = AlphaAgent(game_inst, net)
        # agent.net = torch.load(checkpoints_path +"/checkpoint_10.pt")
        mp.set_start_method('spawn') 
        agent.net.share_memory()
        agent.losses = mp.Manager().list(agent.losses)
        thinkingTime = 40
        c_init = 1.25
        c_base = 19652
        dir_noise = 0.3
        eps = 0.25
        minibatch = 128

        torch.save(agent.net, checkpoints_path + "/challenger.pt")

        

        barrier = mp.Barrier(numCPUs)
        results = mp.Manager().dict()
        iterations = numGames//eval_frequency
        evalGamesPerWorkerPerIter = evalGames//numCPUs
        evalThinkingTime = 40

       
        work = [mp.Process(target=wrapper,args = (checkpoints_path,checkpoints_path + "/challenger.pt",agent,iterations,eval_frequency//numCPUs,thinkingTime,c_init,c_base,dir_noise,eps,barrier,minibatch,evalGamesPerWorkerPerIter,evalThinkingTime,results,i,numGPUs)) for i in range(numCPUs)]

        for p in work:
            p.start()

        
        for p in work:
            p.join()

        torch.save(agent.net, checkpoints_path + "/final_version.pt")

def wrapper(ckpt_path, selfplay_agent_path, train_agent, iterations, local_eval_frequency, thinking_time, c_init,c_base, dir_noise, eps, barrier, minibatch, evalGames, evalThinkingTime, results, idx, numGPUs):
    for i in range(iterations):
        # obtain a copy for selfplay
        dev = 'cuda:' + str(idx % numGPUs)
        selfplay_agent = copy.copy(train_agent)
        selfplay_agent.env.p0_wins = 0
        selfplay_agent.env.p1_wins = 0
        selfplay_agent.env.draws = 0


        selfplay_agent.net = torch.load(selfplay_agent_path)
        # move the transition tuples to the training agent
        selfplay_agent.selfplay(local_eval_frequency, thinking_time, c_init, c_base, dir_noise, eps)


        train_agent.mem.memory = selfplay_agent.mem.memory
        train_agent.train(50, minibatch)

        # wait until all processes finished training the network before evaluating it
        barrier.wait()
        # print(len(train_agent.mem))


        train_agent.evaluate(selfplay_agent.net, evalGames, evalThinkingTime, results, idx)
        
        # wait until all processes finished evaluating, so that one process can collect the results
        barrier.wait()

        if idx == 0:
            torch.save(train_agent.net, ckpt_path + "/checkpoint_" + str(i) + ".pt")
            print(np.mean(train_agent.losses))
            agentWins = 0
            draws = 0
            challengerWins = 0
            for j in range(len(results)):
                agentWins += results[j][0]
                draws += results[j][1]
                challengerWins += results[j][2]

            print("agent's wins: " + str(agentWins) + ", draws: " + str(draws) + ", challenger's wins: " + str(challengerWins))
            if (agentWins)/(agentWins + challengerWins + 1e-6) > 0.55:
                print("new challenger! checkpoint: " + str(i))
                torch.save(train_agent.net, selfplay_agent_path)
            else:
                train_agent.net.load_state_dict(selfplay_agent.net.state_dict())

        # wait until the selfplay & training agents for the next iteration are chosen and saved
        barrier.wait()
            
            
def main():
    checkersNet = neuralNets.CheckersNN(11)
    tictactoeNet = neuralNets.TicTacToeNN(1)
    envTic = envTicTacToe.Env()
    envCheck = env.Env()
    # trainDistributed(tictactoeNet, envTic, 20, 1, 10000, 1000, 100, "checkpoints_tictactoe")
    trainDistributed(checkersNet, envCheck, 10, 1, 10000, 500, 100, "checkpoints_checkers")

main()
