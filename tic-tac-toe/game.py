import env2
from sarsa import Sarsa
import numpy as np
# import benchmark
from tqdm import tqdm
from player0 import Random_Player
from deepQ import deepQNet, DQN
import torch
import deepQ2

def trainSarsa():
    game_inst = env2.Env()
    p0,p1 = Sarsa(0.5, 0.1, 1), Sarsa(0.5, 0.1, 1)
    r_0=[]

    
    if __name__ == '__trainSarsa__':
        for i in tqdm(range(11000)):
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

                    

            
            if i == 10000:
                print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))
                

                # p0.eps = 0
                # p0.alpha = 0
                p1.eps = 0
                p1.alpha = 0
            
    

           
    
    
    print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))

    trained = Sarsa(0,0,0)
    trained.act_space = dict(**p0.act_space, **p1.act_space)

    return trained

def trainDeepQN():
    game_inst = env2.Env()
    p0,p1 = DQN(0.5, 0.1, 1), DQN(0.5, 0.1, 1)
    r_0=[]

    
    if __name__ == '__main__':
        for i in tqdm(range(510000)):
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

                    

            
            if i == 500000:
                print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))
                

                p0.eps = 0
                p0.alpha = 0
                p1.eps = 0
                p1.alpha = 0
                p0.behavior_net.load_state_dict(p0.target_net.state_dict())
                p1.behavior_net.load_state_dict(p1.target_net.state_dict())

            
    

           
    # benchmark.cumulative_reward(r_0)
    
    print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def trainDeepQN2():
    game_inst = env2.Env()
    p0,p1 = deepQ2.DQNAgent(0.5, 0.1, 0.1, 1e-4,10), deepQ2.DQNAgent(0.5, 0.1, 0.1, 1e-4,10)
    r_0=[]

    
    if __name__ == '__main__':
        for i in tqdm(range(11000)):
            game_ended = False
            game_inst.reset()
            p0.curr_state = None
            p1.curr_state = None
            updateNet = False
            R=0
            R_2=0
            A_p=None
            A_2p=None
            if i % 100 == 0:
                 updateNet = True
            
            while True:
                    legal, S_prime, isTerminated = game_inst.get_obs()

                    state_reshp = torch.tensor(S_prime, device=device, dtype=torch.float32).flatten()
                    mask = torch.tensor([state_reshp[i] == 0 for i in range(len(state_reshp))], device=device)


                    if game_inst.turn == -1:
                        A_p = p0.step_agent(A_p, state_reshp, R, mask, isTerminated, updateNet)

                        if A_p is not None:
                            R = game_inst.step_env(np.unravel_index(A_p, S_prime.shape))
                        else:
                            R = game_inst.step_env(None)
                        if R == 1: #A_p was the winning action, modify the last reward of p1 accordingly
                             R_2 = -1
                    else:
                        A_2p = p1.step_agent(A_2p, state_reshp, R_2, mask, isTerminated, updateNet)
                        if A_2p is not None:
                            R_2 = game_inst.step_env(np.unravel_index(A_2p, S_prime.shape))
                        else:
                            R_2 = game_inst.step_env(None)
                        if R_2 == 1:  #A_2p was the winning action, modify the last reward of p0 accordingly
                             R = -1

                    #ensure the loop iterates two more times after the game is finished, so that both players do their last updates
                    if game_ended:
                         break
                    if isTerminated:
                        #  print(game_inst.render())
                         r_0.append(R)
                         game_ended = True

            if i % 100 == 0:
                 updateNet = False

    

            
            if i == 10000:
                print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))
                
                # p0.eps = 0
                # p0.alpha = 0
                # p1.eps = 0
                # p1.alpha = 0
                # p0.behavior_net.load_state_dict(p0.target_net.state_dict())
                # p1.behavior_net.load_state_dict(p1.target_net.state_dict())

    
    print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))



def main():
     trainDeepQN2()

main()