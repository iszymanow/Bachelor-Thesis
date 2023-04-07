import numpy as np

class player:
    """
    class implements a dummy player which just plays random moves
    """

    def __init__(self):
        self.sign = -1
        self.last_action = [-1,-1]
        

    def move(legal_moves, state):
        index = np.random.randint(len(legal_moves))
        return legal_moves[index]

    def reward(reward):
        pass
