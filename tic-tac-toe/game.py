import env2
from sarsa import Sarsa
import numpy as np
from tqdm import tqdm
from player0 import DummyPlayer
import torch
import deepQN
import matplotlib.pyplot as plt


# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
TAU = 1e-6


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
def trainDeepQN():
    game_inst = env2.Env()
    p0 = deepQN.DQNAgent(in_obs=27, 
                              out_actions=9, 
                              alpha=1, 
                              gamma=0.999, 
                              eps_start=1, 
                              eps_end = 0.01, 
                              eps_decay = 1000, 
                              opt_lr=1e-4, 
                              batch_size=512, 
                              tau=TAU,
                              softUpdates=True)
    
    p1 = DummyPlayer()
    
    p0_curr_wins = 0
    p1_curr_wins = 0
    curr_draws = 0

    
    if __name__ == '__main__':
        for i in tqdm(range(1,100001)):
            game_ended = False
            game_inst.reset()
            p0.curr_state_p0 = None
            p0.curr_state_p1 = None

            R=0
            R_2=0
            A_p=None
            A_2p=None

            
            while True:
                    legal, S_prime, isTerminated = game_inst.get_obs()

                    state_reshp = torch.tensor(S_prime, device=device, dtype=torch.float32).flatten()
                    mask = torch.tensor([state_reshp[i] == 0 for i in range(len(state_reshp))], device=device)

                    

                    if game_inst.turn == -1:
                        state_reshp = oneHot(state_reshp).flatten() if not isTerminated else None
                        A_p = p0.step_agent(A_p, state_reshp, R, mask, isTerminated, -1)
                        if A_p is not None:
                            R = game_inst.step_env(np.unravel_index(A_p.cpu().squeeze(), S_prime.shape))
                        else:
                            R = game_inst.step_env(None)
                        if R > 0: #A_p was the winning action, modify the last reward of p1 accordingly
                             R_2 = -R
                    else:
                        state_reshp = oneHot(state_reshp * (-1)).flatten() if not isTerminated else None
                        A_2p = p0.step_agent(A_2p, state_reshp, R_2, mask, isTerminated, 1)
                        # A_2p = p1.step_agent(mask, isTerminated)
                        if A_2p is not None:
                            R_2 = game_inst.step_env(np.unravel_index(A_2p.cpu().squeeze(), S_prime.shape))
                        else:
                            R_2 = game_inst.step_env(None)
                        if R_2 > 0:  #A_2p was the winning action, modify the last reward of p0 accordingly
                             R = -R_2
                    # print(A_p, A_2p)

                    # print(game_inst.render())

                    #ensure the loop iterates two more times after the game is finished, so that both players do their last updates
                    if game_ended:
                         break
                    if isTerminated:
                        #  print(game_inst.render())
                        game_ended = True
                        if R > 0:
                            p0_curr_wins += 1
                        elif R < 0:
                            p1_curr_wins += 1
                        else:
                            curr_draws += 1

            batch_size = 1000
            if i % batch_size == 0:
                p0_batch_win_ratio = p0_curr_wins / batch_size * 100
                p1_batch_win_ratio = p1_curr_wins / batch_size * 100
                draws_batch_ratio = curr_draws / batch_size * 100

                p0_overall_win_ratio = game_inst.p0_wins / i * 100
                p1_overall_win_ratio = game_inst.p1_wins / i * 100
                overall_draws_ratio = game_inst.draws / i * 100
                
                print('TRAIN: last p0`s win ratio: %2.2f, last p1`s win ratio: %2.2f, last draws` ratio: %2.2f' % (p0_batch_win_ratio, p1_batch_win_ratio, draws_batch_ratio))
                print('TRAIN: overall p0`s win ratio: %2.2f, overall p1`s win ratio: %2.2f, overall draws` ratio: %2.2f' % (p0_overall_win_ratio, p1_overall_win_ratio, overall_draws_ratio))

                print("TRAIN: player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))

                plt.plot(p0.losses)
                plt.title("Player0 loss progression")
                plt.xlabel('Number of optimization steps')
                plt.ylabel("loss value")
                plt.savefig("networksAdam/losses_after_" + str(i))

                plt.close()
                p0_curr_wins = 0
                p1_curr_wins = 0
                curr_draws = 0

                if i % (5*batch_size) == 0:
                     testPLay(p0.behavior_net.state_dict(), p0.behavior_net.state_dict(), 27, 9, i)



    torch.save(p0.target_net.state_dict(), 'networksAdam/DQNplayer0.pt')

    print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))
    

def testPLay(net0, net1, in_obs, out_actions, i):


    p0 = deepQN.DQNPlayer(in_obs, out_actions, net0)
    p1 = deepQN.DQNPlayer(in_obs, out_actions, net1)

    game_inst = env2.Env()

    p0_wins,p1_wins,draws=[],[],[]
    if __name__ == '__main__':
        for i in range(1,1000):
            game_ended = False
            game_inst.reset()

            R=0
            R_2=0
            A_p=None
            A_2p=None

            
            while True:
                    legal, S_prime, isTerminated = game_inst.get_obs()

                    state_reshp = torch.tensor(S_prime, device=device, dtype=torch.float32).flatten()
                    mask = torch.tensor([state_reshp[i] == 0 for i in range(len(state_reshp))], device=device)

                    if game_inst.turn == -1:
                        state_reshp = oneHot(state_reshp).flatten()
                        A_p = p0.step_agent(A_p, state_reshp, R, mask, isTerminated)
                        if A_p is not None:
                            R = game_inst.step_env(np.unravel_index(A_p.cpu().squeeze(), S_prime.shape))
                        else:
                            R = game_inst.step_env(None)
                        if R > 0: #A_p was the winning action, modify the last reward of p1 accordingly
                             R_2 = -R
                    else:
                        state_reshp = oneHot(state_reshp * (-1)).flatten() 
                        A_2p = p1.step_agent(A_2p, state_reshp, R_2, mask, isTerminated)
                        # A_2p = p1.step_agent(mask, isTerminated)
                        if A_2p is not None:
                            R_2 = game_inst.step_env(np.unravel_index(A_2p.cpu().squeeze(), S_prime.shape))
                        else:
                            R_2 = game_inst.step_env(None)
                        if R_2 > 0:  #A_2p was the winning action, modify the last reward of p0 accordingly
                             R = -R
                    # print(game_inst.render())

                    #ensure the loop iterates two more times after the game is finished, so that both players do their last updates
                    if game_ended:
                         break
                    if isTerminated:
                         game_ended = True
            
            p0_wins.append(game_inst.p0_wins)
            p1_wins.append(game_inst.p1_wins)
            draws.append(game_inst.draws)

          
 

    print("EVAL: player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))
    plt.plot(p0_wins, label='p0 wins')
    plt.plot(p1_wins, label='p1 wins')
    plt.plot(draws, label='draws')
    # plt.legend()
    plt.title("Evaluation of target networks after " + str(i) + "iterations")
    
    plt.savefig("evaluation/results_after_" + str(i))   
    plt.close() 


def main():
    path0 = 'networks/DQNplayer0.pt'
    path1 = 'networks/DQNplayer1.pt'
    # testPLay(path0, path1)

    trainDeepQN()

main()