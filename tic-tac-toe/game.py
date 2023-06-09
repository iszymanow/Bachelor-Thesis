import env2
from sarsa import Sarsa
import numpy as np
from tqdm import tqdm
import player0
import torch
import deepQN
import matplotlib.pyplot as plt
import random


# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'


def trainSarsa():
    game_inst = env2.Env()
    p0,p1 = Sarsa(0.5, 0.1, 1), Sarsa(0.5, 0.1, 1)
    r_0=[]

    
    if __name__ == '__trainSarsa__':
        for i in tqdm(range(1100)):
            game_ended = False
            game_inst.reset()
            
            legal, S, isTerminated0 = game_inst.get_obs()
            if game_inst.turn == -1:
                A = p0.initialize_sarsa(legal, S)
                R = game_inst.step_env(A)
                

                legal, S, isTerminated1 = game_inst.get_obs()
                A_2 = p1.initialize_sarsa(legal, S) 
                R_2 = game_inst.step_env(A_2)
                
            else:
                legal, S, isTerminated1 = game_inst.get_obs()
                A_2 = p1.initialize_sarsa(legal, S) 
                R_2 = game_inst.step_env(A_2)
                

                legal, S, isTerminated0 = game_inst.get_obs()
                A = p0.initialize_sarsa(legal, S)
                R = game_inst.step_env(A)
                




            while True:
                    legal, S_prime, isTerminated = game_inst.get_obs()

                    if game_inst.turn == -1:
                        A_p = p0.step_sarsa(R, S_prime, legal, isTerminated)
                        R = game_inst.step_env(A_p)
                        if R == 1: #A_p was the winning action, modify the last reward of p1 accordingly
                             R_2 = -1
                    else:
                        A_2p = p1.step_sarsa(R_2, S_prime, legal, isTerminated)
                        R_2 = game_inst.step_env(A_2p)
                        if R_2 == 1:  #A_2p was the winning action, modify the last reward of p0 accordingly
                             R = -1

                    #ensure the loop iterates two more times after the game is finished, so that both players do their last updates
                    if game_ended:
                         break
                    if isTerminated:
                         r_0.append(R)
                         game_ended = True

                    

            
            if i == 1000:
                print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))
                

                # p0.eps = 0
                # p0.alpha = 0
                p1.eps = 0
                p1.alpha = 0
            
    

           
    
    
    print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))

    trained = Sarsa(0,0,0)
    trained.act_space = dict(**p0.act_space, **p1.act_space)

    return trained

    game_inst = env2.Env()
    p0,p1 = DQN(0.5, 0.1, 1), DQN(0.5, 0.1, 1)
    r_0=[]

    
    if __name__ == '__main__':
        for i in tqdm(range(10000)):
            game_ended = False
            game_inst.reset()
            R=0
            R_2=0
            A_p=None
            A_2p=None
            
            # print("running")
            while True:
                    legal, S_prime, isTerminated = game_inst.get_obs()
                    state_reshp = S_prime.reshape((-1,))
                    mask = [(state_reshp[i] != 0) for i in range(len(state_reshp))]

                    if game_inst.turn == -1:
                        A_p = p0.step_agent(A_p, R, torch.Tensor(S_prime), legal, mask, isTerminated)
                        R = game_inst.step_env(A_p)
                        if R == 1: #A_p was the winning action, modify the last reward of p1 accordingly
                             R_2 = -1
                    else:
                        A_2p = p1.step_agent(A_2p, R, torch.Tensor(S_prime), legal, mask, isTerminated)
                        R_2 = game_inst.step_env(A_2p)
                        if R_2 == 1:  #A_2p was the winning action, modify the last reward of p0 accordingly
                             R = -1

                    #ensure the loop iterates two more times after the game is finished, so that both players do their last updates
                    if game_ended:
                         break
                    if isTerminated:
                         r_0.append(R)
                         game_ended = True

                    

            
            if i == 11000:
                print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))
                

                p0.eps = 0
                p0.alpha = 0
                p1.eps = 0
                p1.alpha = 0
                p0.behavior_net.load_state_dict(p0.target_net.state_dict())
                p1.behavior_net.load_state_dict(p1.target_net.state_dict())

            
    

           
    # benchmark.cumulative_reward(r_0)
    
    print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))

def oneHot(state):
     encoding = []
     for entry in state:
        if entry == -1:
            encoding.append([1,0,0])
        elif entry == 0:
            encoding.append([0,1,0])
        else:
            encoding.append([0,0,1]) 
     encoding = torch.tensor([encoding], dtype=torch.float, device=device)
     return encoding
def decode(state):
    decoding = []
    for i in range(0,27,3):
        if [state[i], state[i+1], state[i+2]] == [1,0,0]:
            decoding.append(-1)
        elif [state[i], state[i+1], state[i+2]] == [0,1,0]:
            decoding.append(0)
        else:
            decoding.append(1)
    return decoding

#TODO clean up and document the code
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
    game_inst = env2.Env()
    p0 = deepQN.DQNAgent(in_obs=27, 
                        out_actions=9, 
                        alpha=1, 
                        gamma=0.999, 
                        eps_start=1, 
                        eps_end = 0.15, 
                        eps_decay=0.9999, 
                        opt_lr=6e-4, 
                        batch_size=128, 
                        tau=3e-5,
                        softUpdates=True)
    

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
            game_ended = False
            game_inst.reset()
            p0.curr_state_p0 = None
            p0.curr_state_p1 = None
            R=0
            R_2=0
            A_p=None
            A_2p=None

            # the main episode loop
            while True:
                    legal, S_prime, isTerminated = game_inst.get_obs()
                    state_reshp = torch.tensor(S_prime, device=device, dtype=torch.float32).flatten()
                    mask = torch.tensor([state_reshp[i] == 0 for i in range(len(state_reshp))], device=device)

                    if game_inst.turn == -1:
                        # encode the state for the player0 and make the agent choose the action
                        state_reshp = oneHot(state_reshp).flatten() if not isTerminated else None
                        A_p = p0.step_agent(A_p, state_reshp, R, mask, isTerminated, -1)

                        # environment step
                        if A_p is not None:
                            R = game_inst.step_env(np.unravel_index(A_p.cpu().squeeze(), S_prime.shape))
                        else:
                            R = game_inst.step_env(None)

                        #A_p was the winning action, modify the last reward of p1 accordingly
                        if R > 0:
                             R_2 = -R
                    else:
                        # encode the state for the player1 and make the agent choose the action
                        # IMPORTANT: FLIP THE SIGNS OF THE BOARD BEFORE ENCODING THE STATE!!!
                        # It ensures that the agent learns an optimal strategy for both player0 and player1
                        # (idea based on AlphaZero's flipping board towards current player's direction)
                        state_reshp = oneHot(-state_reshp).flatten() if not isTerminated else None
                        A_2p = p0.step_agent(A_2p, state_reshp, R_2, mask, isTerminated, 1)

                        #environment step
                        if A_2p is not None:
                            R_2 = game_inst.step_env(np.unravel_index(A_2p.cpu().squeeze(), S_prime.shape))
                        else:
                            R_2 = game_inst.step_env(None)

                        #A_2p was the winning action, modify the last reward of p0 accordingly
                        if R_2 > 0:  
                             R = -R_2


                    # ensure the loop iterates two more times after the game is finished,
                    # so that last updates are performed for both p0 and p1
                    if game_ended:
                         break
                    if isTerminated:
                        game_ended = True

                        # update the progress tracking variables
                        if R > 0:
                            p0_curr_wins += 1
                        elif R < 0:
                            p1_curr_wins += 1
                        else:
                            curr_draws += 1


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
                    sp_p0, sp_d, sp_p1 = testSelfPlay(p0.target_net.state_dict(), 27, 9)
                    randS_p0, randS_d, randS_p1 = testRandomPlay(p0.target_net.state_dict(), randomStarts=True)
                    randNS_p0, randNS_d,randNS_p1 = testRandomPlay(p0.target_net.state_dict(), randomStarts=False)

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
                    plt.figure(figsize=(20,5))
                    plt.subplot(1,3,1)
                    plt.plot(x_axis, selfplay_p0_wins, label='p0 wins', marker='.')
                    plt.plot(x_axis, selfplay_draws, label='draws', marker='*')
                    plt.plot(x_axis, selfplay_p1_wins, label='p1 wins', marker='d')
                    plt.legend()
                    plt.xlabel("number of training games played")
                    plt.ylabel("number of games with particular outcome during eval. run")
                    plt.title("Self-play results (First move random with p=0.5)")

                    plt.subplot(1,3,2)
                    plt.plot(x_axis, randomS_p0_wins, label='p0 wins', marker='.')
                    plt.plot(x_axis, randomS_draws, label='draws', marker='*')
                    plt.plot(x_axis, randomS_p1_wins, label='p1 wins', marker='d')
                    plt.legend()
                    plt.xlabel("number of training games played")
                    plt.ylabel("number of games with particular outcome during eval. run")
                    plt.title("Results against a random player that plays as first (rand = p0)")

                    plt.subplot(1,3,3)
                    plt.plot(x_axis, randomNS_p0_wins, label='p0 wins', marker='.')
                    plt.plot(x_axis, randomNS_draws, label='draws', marker='*')
                    plt.plot(x_axis, randomNS_p1_wins, label='p1 wins', marker='d')
                    plt.legend()
                    plt.xlabel("number of training games played")
                    plt.ylabel("number of games with particular outcome during eval. run")
                    plt.title("Results against a random player that plays as second (rand = p1)")

                    plt.savefig(result_plots_path + "/results_after_" + str(i) + "_iters")
                    plt.close()

                    # # stacked bar charts
                    # plt.figure(figsize=(20,5)).tight_layout()
                    # plt.subplot(1,3,1)
                    # plt.bar(x_axis, selfplay_p0_wins, label='p0 wins')
                    # plt.bar(x_axis, selfplay_draws, label='draws', bottom=selfplay_p0_wins)
                    # plt.bar(x_axis, selfplay_p1_wins, label='p1 wins', bottom=np.add(selfplay_p0_wins, selfplay_draws))
                    # plt.legend()
                    # plt.xlabel("number of training games played")
                    # plt.ylabel("number of games with particular outcome during eval. run")
                    # plt.title("Self-play results (First move random with p=0.5)")

                    # plt.subplot(1,3,2)
                    # plt.bar(x_axis, randomS_p0_wins, label='rand. player\'s wins')
                    # plt.bar(x_axis, randomS_draws, label='draws', bottom=randomS_p0_wins)
                    # plt.bar(x_axis, randomS_p1_wins, label='agent\'s wins', bottom=np.add(randomS_p0_wins, randomS_draws))
                    # plt.legend()
                    # plt.xlabel("number of training games played")
                    # plt.ylabel("number of games with particular outcome during eval. run")
                    # plt.title("Results against a random player that plays as first")

                    # plt.subplot(1,3,3)
                    # plt.bar(x_axis, randomNS_p0_wins, label='agent\'s wins')
                    # plt.bar(x_axis, randomNS_draws, label='draws', bottom=randomNS_p0_wins)
                    # plt.bar(x_axis, randomNS_p1_wins, label='rand. player\'s wins',  bottom=np.add(randomNS_p0_wins, randomNS_draws))
                    # plt.legend()
                    # plt.xlabel("number of training games played")
                    # plt.ylabel("number of games with particular outcome during eval. run")
                    # plt.title("Results against a random player that plays as second")

                    # plt.savefig(result_plots_path + "/results_after" + str(i) + "_iters")
                    # plt.close()

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
    game_inst = env2.Env()
    player = deepQN.DQNPlayer(in_obs, out_actions, net)

    if __name__ == '__main__':
        # main testing loop
        for i in range(numEpisodes):
            game_inst.reset()    
            while True:
                    randomStart = bool(random.getrandbits(1))
                    legal, S_prime, isTerminated = game_inst.get_obs()
                    
                    if isTerminated: 
                         break

                    state_reshp = torch.tensor(S_prime, device=device, dtype=torch.float32).flatten()
                    mask = torch.tensor([state_reshp[i] == 0 for i in range(len(state_reshp))], device=device)

                    if game_inst.turn == -1:
                        state_reshp = oneHot(state_reshp).flatten()
                        A_p = player.step_agent(state_reshp, mask, randomStart)
                       
                        game_inst.step_env(np.unravel_index(A_p.cpu().squeeze(), S_prime.shape))
                       
                    else:
                        # IMPORTANT: FLIP THE SIGNS OF THE BOARD BEFORE ENCODING THE STATE!!!
                        state_reshp = oneHot(state_reshp * (-1)).flatten() 
                        A_2p = player.step_agent(state_reshp, mask)
                        
                        game_inst.step_env(np.unravel_index(A_2p.cpu().squeeze(), S_prime.shape))


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
    p0 = deepQN.DQNPlayer(27, 9, network)
    p1 = player0.RandomPlayer()
    game_inst = env2.Env()

    # indicator which player is random
    description = ''
    if randomStarts:
         description = 'rand = p0'
    else:
         description = 'rand = p1'

    # main testing loop
    for i in range(numEpisodes):
        game_inst.reset()
        while True:
            legal, S_prime, isTerminated = game_inst.get_obs()
            shp = S_prime.shape
            state_reshp = torch.tensor(S_prime, device=device, dtype=torch.float32).flatten()
            mask = torch.tensor([state_reshp[i] == 0 for i in range(len(state_reshp))], device=device)

            if isTerminated:
                    break
            
            if game_inst.turn == -1:
                if randomStarts:
                    action = np.unravel_index(p1.step_agent(mask), shp)
                else:
                    action = np.unravel_index(p0.step_agent(oneHot(state_reshp).flatten(), mask).squeeze(), shp)                
            else:
                if randomStarts:
                    # IMPORTANT: FLIP THE SIGNS OF THE BOARD BEFORE ENCODING THE STATE!!!
                    action = np.unravel_index(p0.step_agent(oneHot(-state_reshp).flatten(), mask).squeeze(), shp)
                else:
                    action = np.unravel_index(p1.step_agent(mask), shp)

            game_inst.step_env(action)

    print("EVAL (random, " + description + "): player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))

    return game_inst.p0_wins, game_inst.draws, game_inst.p1_wins

def main():
    trainDeepQN(eval_frequency=1000,
                numEpisodes=30000,
                weights_path='finalRun',
                loss_plots_path='finalRun/losses',
                result_plots_path='finalRun/results')
    
    weights = torch.load('finalRun/DQNAgent.pt')
    testRandomPlay(weights, False, 10000)
    testRandomPlay(weights, True, 10000)
    # testSelfPlay(weights, 27, 9, 10000)



main()