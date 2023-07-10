import torch
import torch.multiprocessing as mp
import numpy as np
from alphaAgent import AlphaAgent
import neuralNets
import copy

import env
import envTicTacToe
import sys
import time
from datetime import datetime

# old_stdout = sys.stdout
# log_file = open("wrapper_" + str(datetime.now()) + ".log","w")
# sys.stdout = log_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def trainDistributed(net, env, numCPUs, numGPUs, numGames, eval_frequency, evalGames, checkpoints_path):
    if __name__ == '__main__':
        
        # logging.basicConfig(level=logging.DEBUG, filename="wrapper_logs", filemode="a+", format="%(asctime)-15s %(message)s")
        game_inst = env
        agent = AlphaAgent(game_inst, net)
        agent.net = torch.load("checkpoints_checkers_32k_30sims/checkpoint_15.pt").to(device)
        mp.set_start_method('spawn') 
        agent.net.share_memory()
        agent.losses = mp.Manager().list(agent.losses)
        thinkingTime = 100
        c_init = 1.25
        c_base = 19652
        dir_noise = 0.3
        eps = 0.25
        minibatch = 64

        torch.save(agent.net, checkpoints_path + "/challenger.pt")

        

        barrier = mp.Barrier(numCPUs)
        results = mp.Manager().dict()
        iterations = numGames//eval_frequency
        evalGamesPerWorkerPerIter = evalGames//numCPUs
        evalThinkingTime = 100

        work = [mp.Process(target=wrapper,args = (checkpoints_path,checkpoints_path + "/challenger.pt",agent,iterations,eval_frequency//numCPUs,thinkingTime,c_init,c_base,dir_noise,eps,barrier,minibatch,evalGamesPerWorkerPerIter,evalThinkingTime,results,i,numGPUs)) for i in range(numCPUs)]

        for p in work:
            p.start()

        
        for p in work:
            p.join()

        torch.save(agent.net, checkpoints_path + "/final_version.pt")

        # sys.stdout = old_stdout
        # log_file.close()

def wrapper(ckpt_path, selfplay_agent_path, train_agent, iterations, local_eval_frequency, thinking_time, c_init,c_base, dir_noise, eps, barrier, minibatch, evalGames, evalThinkingTime, results, idx, numGPUs):
    if idx == 0:
        old_stdout = sys.stdout
        log_file = open("wrapper_" + str(datetime.now()) + ".log","w")
        sys.stdout = log_file
        timestamps = []
    for i in range(iterations):
        if idx == 0:
            start = time.time()
            print(str(datetime.now()) +  " ITERATION " + str(i+1) + ":")
        # obtain a copy for selfplay
        # dev = 'cuda:' + str(idx % numGPUs)
        selfplay_agent = copy.copy(train_agent)
        selfplay_agent.env.p0_wins = 0
        selfplay_agent.env.p1_wins = 0
        selfplay_agent.env.draws = 0


        selfplay_agent.net = torch.load(selfplay_agent_path).to(device)
        # move the transition tuples to the training agent
        # logging.info("Node " + str(idx) + " starting selfplay. Iter: " + str(i) + "/" + str(iterations))
        selfplay_agent.selfplay(local_eval_frequency, thinking_time, c_init, c_base, dir_noise, eps)
        # logging.info("Node " + str(idx) + " finished selfplay. Iter: " + str(i) + "/" + str(iterations))


        train_agent.mem.memory = selfplay_agent.mem.memory
        # logging.info("Node " + str(idx) + " starting training. Iter: " + str(i) + "/" + str(iterations))

        train_agent.train(100, minibatch)
        # logging.info("Node " + str(idx) + " finished training. Iter: " + str(i) + "/" + str(iterations))

        # wait until all processes finished training the network before evaluating it
        barrier.wait()
        # print(len(train_agent.mem))

        # logging.info("Node " + str(idx) + " starting evaluation. Iter: " + str(i) + "/" + str(iterations))

        train_agent.evaluate(selfplay_agent.net, evalGames, evalThinkingTime, results, idx)
        # logging.info("Node " + str(idx) + " finished evaluation. Iter: " + str(i) + "/" + str(iterations))

        # wait until all processes finished evaluating, so that one process can collect the results
        barrier.wait()

        if idx == 0:
            torch.save(train_agent.net, ckpt_path + "/checkpoint_" + str(i) + ".pt")
            print("Training loss mean = " + str(np.mean(train_agent.losses)))
            agentWins = 0
            draws = 0
            challengerWins = 0
            for j in range(len(results)):
                agentWins += results[j][0]
                draws += results[j][1]
                challengerWins += results[j][2]

            print("EVAL: agent's wins: " + str(agentWins) + ", draws: " + str(draws) + ", challenger's wins: " + str(challengerWins))
            if (agentWins)/(agentWins + challengerWins + 1e-6) > 0.55:
                print("EVAL: NEW CHALLENGER! FILE checkpoint_" + str(i))
                torch.save(train_agent.net, selfplay_agent_path)
            else:
                print("EVAL: Old challenger remains.")
                train_agent.net.load_state_dict(selfplay_agent.net.state_dict())
            
            end = time.time()
            elapsed = end - start
            timestamps.append(end-start)
            print("Iter. " + str(i+1) + "/" + str(iterations) + " duration: "  + str(elapsed/60) + "min. Mean iter. duration: " + str(np.mean(timestamps)/60) + "min.\n")

        # wait until the selfplay & training agents for the next iteration are chosen and saved
        barrier.wait()
    if idx == 0:
        sys.stdout = old_stdout
        log_file.close()
            
            
def main():
    checkersNet = neuralNets.CheckersNN(15)
    # tictactoeNet = neuralNets.TicTacToeNN(5)
    # envTic = envTicTacToe.Env()
    envCheck = env.Env()
    # trainDistributed(tictactoeNet, envTic, 10, 1, 100, 100, 100, "checkpoints_tictactoe")
    trainDistributed(checkersNet, envCheck, 5, 1, 10000, 1000, 100, "checkpoints_checkers")

main()
