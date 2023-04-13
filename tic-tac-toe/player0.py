import numpy as np

class Random_Player:
    """
    class that implements a dummy player which just plays random moves
    """
    def __init__(self, sym, env):
        self.sym = sym
        self.env = env
        self.last_move = None
        


    def move(self, state, legal):
        """
        the function which takes the current state, as well as the legal moves and
        chooses a random move out of the available ones.
        If the game is finished, it returns a tuple (-1,-1).
        """
        if self.env.isTerminated(state, legal): #if the game is already finished, just return a dummy output
            return (-1,-1)
        else: # choose a random move from the available ones
            index = np.random.randint(len(legal))
            return legal[index]


    
    def play(self):
        """
        the playing function of the algorithm. It makes use of the environment's condition variable,
        to synchronize with the other player. The player runs the loop until the game terminates"""
        while True:
            with self.env.cv:
                while self.env.turn != self.sym:
                    self.env.cv.wait()

                legal, state = self.env.get_obs()
                self.last_move = self.move(state, legal)
                self.env.action(self.last_move)

                self.env.cv.notify()
            
            if self.last_move == (-1,-1):
                break







