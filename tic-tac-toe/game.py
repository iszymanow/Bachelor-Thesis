import env2
from sarsa import Sarsa
import threading
import numpy as np
import matplotlib.pyplot as plt

def main():
    game_inst = env2.Env()
    p2,p3 = Sarsa(-1, game_inst, 0.2, 0.1, 1), Sarsa(1, game_inst, 0.2, 0.1, 1)
    
    # wins=[]
    # loses=[]
    # draws=[]
    
    if __name__ == '__main__':
        for i in range(10000):
            print("game:",str(i+1) + "/10000")
            
            p = threading.Thread(target=p2.play)
            q = threading.Thread(target=p3.play)
            p.start()
            q.start()

            p.join()
            q.join()


            game_inst.reset()
            # wins.append(game_inst.p1_wins)
            # loses.append(game_inst.p0_wins)
            # draws.append(game_inst.draws)
    
    
    print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))


main()