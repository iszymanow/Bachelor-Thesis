import numpy as np
import torch
import copy

class Env:
    """
    The environment implementation for Tic-Tac-Toe.
    
    state representation: 3x3 numpy array. Empty fields are filled with 0s,
    player's 0 fields are filled with -1s and player's 1 fields are filled with 1s

    action representation: a tuple (x,y), which indicates the entry in the state array
    (i.e. row x, col. y) chosen by the player

    reward system:
    * +1 for a win
    * -1 for a loss
    * 0 for a draw or a non-terminating move
    """

    def __init__(self):
        """
        initialize the game environment, along with results' info 
        and a condition variable for efficient interactions with the two players
        """
        #state variables
        # self.turn = np.random.choice([-1,1])
        self.turn = -1
        self.state = np.zeros(9, dtype=np.int8)
        self.draw = False
        self.done = 3

        #results info
        self.p0_wins= 0
        self.p1_wins= 0
        self.draws=0

    def __str__(self) -> str:
        return str(self.state)
    
    def getBoard(self):
        cp = self.state.copy()
        if self.turn == 1:
            cp *= -1
        return cp
    

    def encodeObs(self):
        state = self.getBoard().reshape((3,3,))
        encoded = np.zeros([2,3,3])

        encoded[0,:,:][state==-1] = 1
        encoded[1,:,:][state==1] = 1

        return torch.tensor(encoded, dtype=torch.float)
    
    def save(self):
        return copy.deepcopy(self)


    def reset(self):
        """
        reset the environment, useful when one wants to play multiple games in a row
        """
        curr_p0_wins = self.p0_wins
        curr_p1_wins = self.p1_wins
        curr_draws = self.draws

        self.__init__()
        self.p0_wins = curr_p0_wins
        self.p1_wins = curr_p1_wins
        self.draws = curr_draws
        
    def get_mask(self, player):
        mask = np.array([True if i == 0 else False for i in self.state])

        return mask


    def get_obs(self):
        """
        returns an array of legal actions, the current state of the game and a boolean indicating whether the game is finished
        """
        state = self.getBoard()
        mask = np.array([True if i == 0 else False for i in state])
        done = self.isTerminated(mask) !=3 
        
        return state, mask, done

        

    def isTerminated(self,mask):
        """
        Check if the game is finished,
        returns True if either player wins or if there is a draw, otherwise returns False
        """
        #check both rows and columns
        state = self.state.reshape((3,3))

        for i in range(3):
            if np.abs(np.sum(state[i])) == 3 or np.abs(np.sum(state[:,i])) == 3:
                self.done = -1
                if self.turn == -1:
                    self.p1_wins += 1
                else:
                    self.p0_wins += 1
                return self.done
            

        
        #check the diagonals
        if np.abs(np.trace(state)) == 3 or np.abs(np.trace(np.fliplr(state))) == 3:
            self.done = -1
            if self.turn == -1:
                self.p1_wins += 1
            else:
                self.p0_wins += 1
            return self.done

        #check if there are any legal moves left
        if not mask.any():
            self.draw = True
            self.draws += 1
            self.done = 0
        
        return self.done



    def step_env(self, move):
        """
        function called by the players interchangeably during the game. 
        It takes a move as an argument and changes the game's state according to the given move.
        The return value is the reward for the player that made the move.
        """
        reward0, reward1 = 0,0

   

        # check if the game is already finished - don't let the player change the state if that's the case
        if self.done!=3: 
            # self.render() #print the end-state of the game
            if self.turn == -1:
                reward0 = self.done
                reward1 = -self.done
            else:
                reward0 = -self.done
                reward1 = self.done
        else: # game not finished, let the player change the state
            self.state[move] = self.turn
        
        self.turn *= (-1)

        return reward0, reward1



    def render(self):
        """
        output a graphical representation of the current state of the game
        """
        state = self.getBoard().reshape((3,3,))
        board = "\n"
        for row in state:
            for col in row:
                entry = col
                if entry == -1:
                    board += " O |"
                elif entry == 1:
                    board += " X |"
                else:
                    board += "   |"

            board = board[0:(len(board) - 2)]
            board += "\n---+---+---\n"
        
        print(board[0:(len(board) - 12)])