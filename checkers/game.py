import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import env
import random
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import multiprocessing, concurrent.futures
import torch.multiprocessing as mp

#pc:
sys.path.append('/home/igor/Bachelor-Thesis/tic-tac-toe/')

#laptop
# sys.path.append('/users/igor/Bachelor-Thesis/tic-tac-toe')
import deepQN, player0
import os
import time


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'

def encodeObs(state, repetitions, nonprogress, turn):
    if turn == 1:
        state *= -1
    encoded = np.zeros((64,5), dtype=int)
    encoded[np.arange(64), state + 2] = 1

    return torch.tensor(encoded, dtype=torch.float).flatten()

def play(agent, game_inst, numEpisodes):
    for i in tqdm(range(1,numEpisodes + 1)):
            # reset the state variables of the previous game before the next starts 
            game_inst.reset()
            curr_state_p0 = None
            curr_state_p1 = None
            R=0
            R_2=0
            A_p=None
            A_2p=None

            # the main episode loop
            while True:
                    S_prime, mask, isTerminated, repetitions, nonprogress = game_inst.get_obs()
                    mask = mask.flatten()

                    if isTerminated:
                        R,R_2 = game_inst.step_env(None)
                        agent.step_agent(curr_state_p0, A_p, torch.zeros([64*5]), R, mask, isTerminated, -1)
                        agent.step_agent(curr_state_p1, A_2p, torch.zeros([64*5]), R_2, mask, isTerminated, 1)                       
                        break

                    S_prime = encodeObs(S_prime, repetitions, nonprogress, game_inst.turn)
                    
                    if game_inst.turn == -1:
                        A_p = agent.step_agent(curr_state_p0, A_p, S_prime, R, mask, isTerminated, -1)
                        curr_state_p0 = S_prime.clone()
                        # environment step
                        R,R_2 = game_inst.step_env(A_p.to(device))

                    else:
                        A_2p = agent.step_agent(curr_state_p1,A_2p, S_prime, R_2, mask, isTerminated, 1)
                        curr_state_p1 = S_prime.clone()
                        #environment step
                        R,R_2 = game_inst.step_env(A_2p.to(device))
    
    return game_inst.p0_wins, game_inst.draws, game_inst.p1_wins

def trainDistributed(eval_frequency, numEpisodes, weights_path='', loss_plots_path='', result_plots_path=''):
    agent = deepQN.DQNAgent(in_obs=64*5, 
                    out_actions=512, 
                    alpha=1, 
                    gamma=0.999, 
                    eps_start=1, 
                    eps_end = 0.1, 
                    eps_decay=0.9999, 
                    opt_lr=5e-4, 
                    batch_size=256, 
                    tau=5e-6,
                    softUpdates=True)
    
    mp_agent = agent.share_memory()
    randomS_p0_wins, randomS_draws, randomS_p1_wins = [],[],[]
    randomNS_p0_wins, randomNS_draws, randomNS_p1_wins = [],[],[]
    selfplay_p0_wins, selfplay_draws, selfplay_p1_wins = [],[],[]
    
    if __name__ == '__main__':

        

        mp.set_start_method('spawn')
        game_insts = [env.Env() for i in range(10)]
        eval_rounds = numEpisodes//eval_frequency

        for i in range(eval_rounds):
            work = [mp.Process(target=play, args=(mp_agent, game_inst, eval_frequency//10,)) for game_inst in game_insts]
            for p in work:
                p.start()
            
            for p in work:
                p.join()

            # loss progression
            plt.figure(figsize=(10,5))
            plt.plot(agent.losses)
            plt.title("Agent's loss progression")
            plt.xlabel('Number of optimization steps')
            plt.ylabel("loss value")
            plt.savefig(loss_plots_path + "/losses_after_" + str(eval_frequency *(i + 1)) + "_iters")
            plt.close()

            # target network evaluation
            x_axis = [str(x) for x in range(eval_frequency, (i+1)*eval_frequency+1, eval_frequency)]
            sp_p0, sp_d, sp_p1 = testSelfPlay(agent.target_net.state_dict(), 64*5, 512, numEpisodes=10)
            randS_p0, randS_d, randS_p1 = testRandomPlay(agent.target_net.state_dict(), randomStarts=True, numEpisodes=10)
            randNS_p0, randNS_d,randNS_p1 = testRandomPlay(agent.target_net.state_dict(), randomStarts=False, numEpisodes=10)

            selfplay_p0_wins.append(sp_p0)
            selfplay_draws.append(sp_d)
            selfplay_p1_wins.append(sp_p1)

            randomS_p0_wins.append(randS_p0)
            randomS_draws.append(randS_d)
            randomS_p1_wins.append(randS_p1)

            randomNS_p0_wins.append(randNS_p0)
            randomNS_draws.append(randNS_d)
            randomNS_p1_wins.append(randNS_p1)
            # lines/markers plots
            # plt.figure(figsize=(20,5))
            # plt.subplot(1,3,1)
            # plt.plot(x_axis, selfplay_p0_wins, label='p0 wins', marker='.')
            # plt.plot(x_axis, selfplay_draws, label='draws', marker='*')
            # plt.plot(x_axis, selfplay_p1_wins, label='p1 wins', marker='d')
            # plt.legend()
            # plt.xlabel("number of training games played")
            # plt.ylabel("number of games with particular outcome during eval. run")
            # plt.title("Self-play results (First move random with p=0.5)")

            # plt.subplot(1,3,2)
            # plt.plot(x_axis, randomS_p0_wins, label='p0 wins', marker='.')
            # plt.plot(x_axis, randomS_draws, label='draws', marker='*')
            # plt.plot(x_axis, randomS_p1_wins, label='p1 wins', marker='d')
            # plt.legend()
            # plt.xlabel("number of training games played")
            # plt.ylabel("number of games with particular outcome during eval. run")
            # plt.title("Results against a random player that plays as first (rand = p0)")

            # plt.subplot(1,3,3)
            # plt.plot(x_axis, randomNS_p0_wins, label='p0 wins', marker='.')
            # plt.plot(x_axis, randomNS_draws, label='draws', marker='*')
            # plt.plot(x_axis, randomNS_p1_wins, label='p1 wins', marker='d')
            # plt.legend()
            # plt.xlabel("number of training games played")
            # plt.ylabel("number of games with particular outcome during eval. run")
            # plt.title("Results against a random player that plays as second (rand = p1)")

            # plt.savefig(result_plots_path + "/results_after_" + str(i) + "_iters")
            # plt.close()

            # stacked bar charts
            plt.figure(figsize=(20,5)).tight_layout()
            plt.subplot(1,3,1)
            plt.bar(x_axis, selfplay_p0_wins, label='p0 wins')
            plt.bar(x_axis, selfplay_draws, label='draws', bottom=selfplay_p0_wins)
            plt.bar(x_axis, selfplay_p1_wins, label='p1 wins', bottom=np.add(selfplay_p0_wins, selfplay_draws))
            plt.legend()
            plt.xlabel("number of training games played")
            plt.ylabel("number of games with particular outcome during eval. run")
            plt.title("Self-play results (First move random with p=0.5)")

            plt.subplot(1,3,2)
            plt.bar(x_axis, randomS_p0_wins, label='rand. player\'s wins')
            plt.bar(x_axis, randomS_draws, label='draws', bottom=randomS_p0_wins)
            plt.bar(x_axis, randomS_p1_wins, label='agent\'s wins', bottom=np.add(randomS_p0_wins, randomS_draws))
            plt.legend()
            plt.xlabel("number of training games played")
            plt.ylabel("number of games with particular outcome during eval. run")
            plt.title("Results against a random player that plays as first")

            plt.subplot(1,3,3)
            plt.bar(x_axis, randomNS_p0_wins, label='agent\'s wins')
            plt.bar(x_axis, randomNS_draws, label='draws', bottom=randomNS_p0_wins)
            plt.bar(x_axis, randomNS_p1_wins, label='rand. player\'s wins',  bottom=np.add(randomNS_p0_wins, randomNS_draws))
            plt.legend()
            plt.xlabel("number of training games played")
            plt.ylabel("number of games with particular outcome during eval. run")
            plt.title("Results against a random player that plays as second")

            plt.savefig(result_plots_path + "/results_after_" + str((i+1)*eval_frequency) + "_iters")
            plt.close()

            # store the current version of the target network weights
            if i % (10 * eval_frequency):
                    torch.save(agent.target_net.state_dict(), weights_path + '/DQNAgent.pt')


        torch.save(agent.target_net.state_dict(), weights_path + '/DQNAgent.pt')
        print("TRAIN: Training finished, the target weights have been saved at " + weights_path + ".")

def trainDeepQN(eval_frequency, numEpisodes, weights_path='', loss_plots_path='', result_plots_path=''):
    """
    Function used for training the DQN agent. It trains over the specified number of episodes,
    executes evaluation runs periodically and stores the trained network

    Every eval_frequency it provides basic statistics about the overall results
    and results within the last eval_frequency training games.

    Every 5 * eval_frequency it executes an evaluation run of the target network consisting of:
     - 1000 self-play games (where the first move of the game is randomized with prob.=1/2)
     - 1000 games against a randomized player, where the randomized player starts
     - 1000 games against a randomized player, where the tested agent starts
    Afterwards, it plots the loss progression of the agent, as well the evaluation test results.
    It also prints out the evaluation results.

    Every 10 * eval_frequency it stores the current weights of the target network of the agent.

    By the end of the training, the target weights are stored in the specified location

    Args:
        eval_frequency (long): determines the frequency of evaluation runs
        numEpisodes (long): determines the number of episodes the agent is trained on
        weights_path (str, optional): The destination path of the network's weights. Defaults to ''.
        loss_plots_path (str, optional): The destination path of the loss progression plots. Defaults to ''.
        result_plots_path (str, optional): The destination path of the evaluation results plots. Defaults to ''.
    """

    # game environment & agent initialization
    game_inst = env.Env()
    p0 = deepQN.DQNAgent(in_obs=64*5, 
                        out_actions=512, 
                        alpha=1, 
                        gamma=0.999, 
                        eps_start=1, 
                        eps_end = 0.1, 
                        eps_decay=0.9999, 
                        opt_lr=5e-4, 
                        batch_size=256, 
                        tau=5e-6,
                        softUpdates=True)
    
    # weights = torch.load('testRun/DQNAgent30k.pt')
    # p0.behavior_net.load_state_dict(weights)
    # p0.target_net.load_state_dict(weights)
    

    # variables and lists responsible for benchmarking
    p0_curr_wins = 0
    p1_curr_wins = 0
    curr_draws = 0
    randomS_p0_wins, randomS_draws, randomS_p1_wins = [],[],[]
    randomNS_p0_wins, randomNS_draws, randomNS_p1_wins = [],[],[]
    selfplay_p0_wins, selfplay_draws, selfplay_p1_wins = [],[],[]
    
    # the main training loop
    if __name__ == '__main__':
        for i in tqdm(range(1,numEpisodes + 1)):
            # reset the state variables of the previous game before the next starts 
            game_inst.reset()
            curr_state_p0 = None
            curr_state_p1 = None
            R=0
            R_2=0
            A_p=None
            A_2p=None

            # the main episode loop
            while True:
                    S_prime, mask, isTerminated, repetitions, nonprogress = game_inst.get_obs()
                    mask = mask.flatten()

                    if isTerminated:
                        R,R_2 = game_inst.step_env(None)
                        p0.step_agent(curr_state_p0,A_p, torch.zeros([64*5]), R, mask, isTerminated, -1)
                        p0.step_agent(curr_state_p1,A_2p, torch.zeros([64*5]), R_2, mask, isTerminated, 1)

                        # update the progress tracking variables
                        if R > 0:
                            p0_curr_wins += 1
                        elif R < 0:
                            p1_curr_wins += 1
                        else:
                            curr_draws += 1
                        
                        break

                    S_prime = encodeObs(S_prime, repetitions, nonprogress, game_inst.turn)

                    
                    if game_inst.turn == -1:
                        # encode the state for the player0 and make the agent choose the action
                        A_p = p0.step_agent(curr_state_p0,A_p, S_prime, R, mask, isTerminated, -1)
                        curr_state_p0 = S_prime.clone()

                        # environment step
                        R,R_2 = game_inst.step_env(A_p.to(device))

                    else:
                        A_2p = p0.step_agent(curr_state_p1,A_2p, S_prime, R_2, mask, isTerminated, 1)
                        curr_state_p1 = S_prime.clone()
                        #environment step
                        R,R_2 = game_inst.step_env(A_2p.to(device))




                    # game_inst.render(orient=game_inst.turn)
                    # ensure the loop iterates two more times after the game is finished,
                    # so that last updates are performed for both p0 and p1


            # Progress tracking and evaluation part
            if i % eval_frequency == 0:
                p0_batch_win_ratio = p0_curr_wins / eval_frequency * 100
                p1_batch_win_ratio = p1_curr_wins / eval_frequency * 100
                draws_batch_ratio = curr_draws / eval_frequency * 100

                p0_overall_win_ratio = game_inst.p0_wins / i * 100
                p1_overall_win_ratio = game_inst.p1_wins / i * 100
                overall_draws_ratio = game_inst.draws / i * 100
                
                print('TRAIN: last p0`s win ratio: %2.2f, last p1`s win ratio: %2.2f, last draws` ratio: %2.2f' % 
                      (p0_batch_win_ratio, p1_batch_win_ratio, draws_batch_ratio))
                print('TRAIN: overall p0`s win ratio: %2.2f, overall p1`s win ratio: %2.2f, overall draws` ratio: %2.2f' % 
                      (p0_overall_win_ratio, p1_overall_win_ratio, overall_draws_ratio))

                print("TRAIN: player0 wins: " + str(game_inst.p0_wins), 
                    "player1 wins: " + str(game_inst.p1_wins), 
                    "draws: " + str(game_inst.draws))

                # reset the last results for the new batch
                p0_curr_wins = 0
                p1_curr_wins = 0
                curr_draws = 0

                if i % (5*eval_frequency) == 0:
                    # loss progression
                    plt.figure(figsize=(10,5))
                    plt.plot(p0.losses)
                    plt.title("Agent's loss progression")
                    plt.xlabel('Number of optimization steps')
                    plt.ylabel("loss value")
                    plt.savefig(loss_plots_path + "/losses_after_" + str(i) + "_iters")
                    plt.close()

                    # target network evaluation
                    x_axis = [str(x) for x in range(5 * eval_frequency, i+1, 5 * eval_frequency)]
                    sp_p0, sp_d, sp_p1 = testSelfPlay(p0.target_net.state_dict(), 64*5, 512, numEpisodes=1000)
                    randS_p0, randS_d, randS_p1 = testRandomPlay(p0.target_net.state_dict(), randomStarts=True, numEpisodes=1000)
                    randNS_p0, randNS_d,randNS_p1 = testRandomPlay(p0.target_net.state_dict(), randomStarts=False, numEpisodes=1000)

                    selfplay_p0_wins.append(sp_p0)
                    selfplay_draws.append(sp_d)
                    selfplay_p1_wins.append(sp_p1)

                    randomS_p0_wins.append(randS_p0)
                    randomS_draws.append(randS_d)
                    randomS_p1_wins.append(randS_p1)

                    randomNS_p0_wins.append(randNS_p0)
                    randomNS_draws.append(randNS_d)
                    randomNS_p1_wins.append(randNS_p1)

                    # lines/markers plots
                    # plt.figure(figsize=(20,5))
                    # plt.subplot(1,3,1)
                    # plt.plot(x_axis, selfplay_p0_wins, label='p0 wins', marker='.')
                    # plt.plot(x_axis, selfplay_draws, label='draws', marker='*')
                    # plt.plot(x_axis, selfplay_p1_wins, label='p1 wins', marker='d')
                    # plt.legend()
                    # plt.xlabel("number of training games played")
                    # plt.ylabel("number of games with particular outcome during eval. run")
                    # plt.title("Self-play results (First move random with p=0.5)")

                    # plt.subplot(1,3,2)
                    # plt.plot(x_axis, randomS_p0_wins, label='p0 wins', marker='.')
                    # plt.plot(x_axis, randomS_draws, label='draws', marker='*')
                    # plt.plot(x_axis, randomS_p1_wins, label='p1 wins', marker='d')
                    # plt.legend()
                    # plt.xlabel("number of training games played")
                    # plt.ylabel("number of games with particular outcome during eval. run")
                    # plt.title("Results against a random player that plays as first (rand = p0)")

                    # plt.subplot(1,3,3)
                    # plt.plot(x_axis, randomNS_p0_wins, label='p0 wins', marker='.')
                    # plt.plot(x_axis, randomNS_draws, label='draws', marker='*')
                    # plt.plot(x_axis, randomNS_p1_wins, label='p1 wins', marker='d')
                    # plt.legend()
                    # plt.xlabel("number of training games played")
                    # plt.ylabel("number of games with particular outcome during eval. run")
                    # plt.title("Results against a random player that plays as second (rand = p1)")

                    # plt.savefig(result_plots_path + "/results_after_" + str(i) + "_iters")
                    # plt.close()

                    # stacked bar charts
                    plt.figure(figsize=(20,5)).tight_layout()
                    plt.subplot(1,3,1)
                    plt.bar(x_axis, selfplay_p0_wins, label='p0 wins')
                    plt.bar(x_axis, selfplay_draws, label='draws', bottom=selfplay_p0_wins)
                    plt.bar(x_axis, selfplay_p1_wins, label='p1 wins', bottom=np.add(selfplay_p0_wins, selfplay_draws))
                    plt.legend()
                    plt.xlabel("number of training games played")
                    plt.ylabel("number of games with particular outcome during eval. run")
                    plt.title("Self-play results (First move random with p=0.5)")

                    plt.subplot(1,3,2)
                    plt.bar(x_axis, randomS_p0_wins, label='rand. player\'s wins')
                    plt.bar(x_axis, randomS_draws, label='draws', bottom=randomS_p0_wins)
                    plt.bar(x_axis, randomS_p1_wins, label='agent\'s wins', bottom=np.add(randomS_p0_wins, randomS_draws))
                    plt.legend()
                    plt.xlabel("number of training games played")
                    plt.ylabel("number of games with particular outcome during eval. run")
                    plt.title("Results against a random player that plays as first")

                    plt.subplot(1,3,3)
                    plt.bar(x_axis, randomNS_p0_wins, label='agent\'s wins')
                    plt.bar(x_axis, randomNS_draws, label='draws', bottom=randomNS_p0_wins)
                    plt.bar(x_axis, randomNS_p1_wins, label='rand. player\'s wins',  bottom=np.add(randomNS_p0_wins, randomNS_draws))
                    plt.legend()
                    plt.xlabel("number of training games played")
                    plt.ylabel("number of games with particular outcome during eval. run")
                    plt.title("Results against a random player that plays as second")

                    plt.savefig(result_plots_path + "/results_after_" + str(i) + "_iters")
                    plt.close()

            # store the current version of the target network weights
            if i % (10 * eval_frequency):
                    torch.save(p0.target_net.state_dict(), weights_path + '/DQNAgent.pt')


    torch.save(p0.target_net.state_dict(), weights_path + '/DQNAgent.pt')
    print("TRAIN: Training finished, the target weights have been saved at " + weights_path + ".")
    

def testSelfPlay(net, in_obs, out_actions, numEpisodes=1000):
    """
    The function used for evaluation of the trained model. The model plays against itself a specified
    number of times, where the first move of the game is randomized with prob. = 1/2.
    The function returns the results of the games.

    Args:
        net (torch state_dict with layout same as DQNAgent): The weights of the tested neural network
        in_obs (long): the number of input units to the network
        out_actions (long): the number of output units of the network
        numEpisodes (long): the number of games that the model is tested on

    Returns:
        long, long, long: number of player0's wins, number of draws, number of player1's wins
    """

    # initialize the game environment and the tested agent
    game_inst = env.Env()
    player = deepQN.DQNPlayer(in_obs, out_actions, net)

    if __name__ == '__main__':
        # main testing loop
        for i in tqdm(range(numEpisodes)):
            randomStart = bool(random.getrandbits(1))
            game_inst.reset()    
            while True:
                S_prime, mask, isTerminated, repetitions, nonprogress = game_inst.get_obs()

                if isTerminated:
                    break

                S_prime = encodeObs(S_prime, repetitions, nonprogress, game_inst.turn)
                mask = mask.flatten()

                if game_inst.turn == -1:
                    A_p = player.step_agent(S_prime, mask, randomStart)
                    game_inst.step_env(A_p.to(device))
                    
                else:
                    A_2p = player.step_agent(S_prime, mask)
                    game_inst.step_env(A_2p.to(device))


    print("EVAL (selfplay): player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))
    return game_inst.p0_wins, game_inst.draws, game_inst.p1_wins

def testRandomPlay(network, randomStarts, numEpisodes=1000):
    """Function that tests the agent's performance against a fully random player for
    a given  number of episodes.

    Args:
        network (torch state_dict with layout same as the DQNAgent): the weights of the model that we want to evaluate
        randomStarts (Boolean): variable which indicates if the random player moves as first (set to True if yes)
        numEpisodes (int, optional): the number of episodes that we test on. Defaults to 1000.

    Returns:
        long, long, long: number of player0's wins, number of draws, number of player1's wins
    """
    
    # initialize the players and the game env.
    p0 = deepQN.DQNPlayer(64*5, 512, network)
    p1 = player0.RandomPlayer()
    game_inst = env.Env()

    # indicator which player is random
    description = ''
    if randomStarts:
         description = 'rand = p0'
    else:
         description = 'rand = p1'

    # main testing loop
    for i in tqdm(range(numEpisodes)):
        game_inst.reset()
        while True:
            S_prime, mask, isTerminated, repetitions, nonprogress = game_inst.get_obs()
            if isTerminated:
                break

            S_prime = encodeObs(S_prime, repetitions, nonprogress, game_inst.turn)
            mask = mask.flatten()

            if game_inst.turn == -1:
                if randomStarts:
                    action = p1.step_agent(mask)
                else:
                    action = p0.step_agent(S_prime, mask).to(device)
            else:
                if randomStarts:
                    action = p0.step_agent(S_prime, mask).to(device)
                else:
                    action = p1.step_agent(mask)

            game_inst.step_env(action)

    print("EVAL (random, " + description + "): player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))

    return game_inst.p0_wins, game_inst.draws, game_inst.p1_wins

def playerMove(move, orient):
    start = int(move//8)
    action = int(move % 8)
    print("player" + str(orient) + " moved from square no. " + str(start) + ' and chose action ' + str(action))

def manualTestPlay(model):
    p0 = deepQN.DQNPlayer(64*5, 512, model)
    game_inst = env.Env()

    playAgain = True
    while playAgain:
        game_inst.reset()
        playerSign = 0
        while playerSign == 0:
            whoStarts = input("Choose white or black pieces [W/B]: ")
            match whoStarts:
                case 'B': playerSign = -1
                case 'W': playerSign = 1
                case _: print('Invalid input. \n')


        while True:
            S_prime, mask, isTerminated, repetitions, nonprogress = game_inst.get_obs()
            if isTerminated:
                break

            mask = mask.flatten()

            
            if game_inst.turn == playerSign:
                os.system('clear')
                game_inst.render(orient=playerSign, manualAid=True)
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
                                action = torch.tensor(8*move + action, device=device)
                    else:
                        print("No piece on the square or no possible moves with the piece\n")
        
            else:
                S_prime = encodeObs(S_prime, repetitions, nonprogress, game_inst.turn)

                action = p0.step_agent(S_prime.float(), mask)

                if action is not None:
                    action = action.to(device)

            # playerMove(action, orient=game_inst.turn)
            game_inst.step_env(action)
      
        game_inst.render(orient=playerSign)
        again = input('Game finished. Want to play again? [y/n]: ')
        match again:
            case 'y':
                playAgain = True
            case 'n':
                playAgain = False
            case _:
                playAgain = False






def main():
    trainDeepQN(eval_frequency=1000,
                numEpisodes=30000,
                weights_path='testRun',
                loss_plots_path='testRun/losses',
                result_plots_path='testRun/results')
    
    # weights = torch.load('testRun/DQNAgent10k.pt')
    # testRandomPlay(weights, False, 1000)
    # testRandomPlay(weights, True, 1000)
    # testSelfPlay(weights, 64*5, 512, 1000)
    # manualTestPlay(weights)

    # trainDistributed(eval_frequency=3000,
    #             numEpisodes=30000,
    #             weights_path='testRun',
    #             loss_plots_path='testRun/losses',
    #             result_plots_path='testRun/results')



main()