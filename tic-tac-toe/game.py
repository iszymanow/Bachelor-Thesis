import env2
from sarsa import Sarsa
import threading
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from player0 import Random_Player

def main():
    game_inst = env2.Env()
    p0,p1 = Sarsa(0.2, 0.1, 1), Sarsa(0.2, 0.1, 1)

    
    if __name__ == '__main__':
        for i in tqdm(range(21000)):
            game_ended = False
            game_inst.reset()
            

            if game_inst.turn == -1:
                legal, S, isTerminated0 = game_inst.get_obs()
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
                        if R == 1: #A_p was the winning action, modify the last of p1 accordingly
                             R_2 = -1
                    else:
                        A_2p = p1.step_sarsa(R_2, S_prime, legal, isTerminated)
                        R_2 = game_inst.step_env(A_2p)
                        if R_2 == 1:  #A_2p was the winning action, modify the last of p0 accordingly
                             R = -1

                    if game_ended:
                         break
                    if isTerminated:
                         game_ended = True

                    

            
            if i == 20000:
                print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))
                

                p1.eps = 0
                p1.alpha = 0
                
            



           
    
    
    print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))


main()