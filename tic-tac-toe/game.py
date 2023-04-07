from env import Env
from player0 import player
from sarsa import Sarsa_learner
import numpy as np

def main():
    p0, p1 = player, player
    p2 = Sarsa_learner(0.1, 0.2, 1)
    p2.generate_action_space()
    game_inst = Env(p0, p2)

    # for i in range(10000):
    #     game_inst.play()
    #     p2.reset()
    
    
    print("player0 wins: " + str(game_inst.p0_wins), "player1 wins: " + str(game_inst.p1_wins), "draws: " + str(game_inst.draws))


main()