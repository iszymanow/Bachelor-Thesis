import numpy as np

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
        self.state = np.zeros((3,3))
        self.draw = False

        #results info
        self.p0_wins= 0
        self.p1_wins= 0
        self.draws=0



    def reset(self):
        """
        reset the environment, useful when one wants to play multiple games in a row
        """
        self.draw = False
        # self.turn = np.random.choice([-1,1])
        self.turn = -1
        self.state = np.zeros((3,3))
        


    def get_obs(self):
        """
        returns an array of legal actions, the current state of the game and a boolean indicating whether the game is finished
        """
        legal = [(i, j) for i in range(self.state.shape[0]) for j in range(self.state.shape[1]) if self.state[i][j] == 0]
        state = np.copy(self.state)
        isTerminated = self.isTerminated(state, legal)
        
        return legal, state, isTerminated

        

    def isTerminated(self, state, legal):
        """
        Check if the game is finished,
        returns True if either player wins or if there is a draw, otherwise returns False
        """
        #check both rows and columns
        for i in range(3):
            if np.abs(np.sum(state[i])) == 3 or np.abs(np.sum(state[:,i])) == 3:
                return True
        
        #check the diagonals
        if np.abs(np.trace(state)) == 3 or np.abs(np.trace(np.fliplr(state))) == 3:
            return True

        #check if there are any legal moves left
        if legal == []:
            self.draw = True
            return True
        
        return False



    def step_env(self, move):
        """
        function called by the players interchangeably during the game. 
        It takes a move as an argument and changes the game's state according to the given move.
        The return value is the reward for the player that made the move.
        """
        reward = 0

        legal, state, isTerminated = self.get_obs()

        # check if the game is already finished - don't let the player change the state if that's the case
        if isTerminated: 
            # self.render() #print the end-state of the game
            if self.draw:
                reward = 0
            else:
                reward = -1
        else: # game not finished, let the player change the state
            self.state[move[0]][move[1]] = self.turn
            legal, state, isTerminated = self.get_obs()

            if isTerminated:
                # self.render() #print the end-state of the game

                if self.draw:
                    self.draws += 1
                    reward = 0
                else:
                    if self.turn == -1:
                        self.p0_wins +=1
                    else:
                        self.p1_wins += 1
                    reward = 1
        
        self.turn *= (-1)

        return reward



    def render(self):
        """
        output a graphical representation of the current state of the game
        """
        state = self.state
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
        
        return board[0:(len(board) - 12)]