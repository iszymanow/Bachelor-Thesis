import numpy as np

class Env:

    def __init__(self, player0, player1):
        """
        initialize the game environment:
        - specify the two players
        - decide who starts
        - prepare the board
        """
        self.p0 = player0
        self.p1 = player1
        self.turn = np.random.choice([-1,1])
        self.state = np.zeros((3,3))
        self.draw = False

        self.p0_wins= 0
        self.p1_wins= 0
        self.draws=0



    def reset(self):
        self.draw = False
        self.turn = np.random.choice([-1,1])
        self.state = np.zeros((3,3))
        



    def get_obs(self):
        """
        returns who's turn it is, an array of legal actions and the current state of the game
        """
        legal = [(i, j) for i in range(self.state.shape[0]) for j in range(self.state.shape[1]) if self.state[i][j] == 0]
        turn = self.turn
        state = self.state

        return legal, turn, state

        

    def isTerminated(self):
        legal, turn, state = self.get_obs()
        for i in range(3):
            if np.abs(np.sum(state[i])) == 3 or np.abs(np.sum(state[:,i])) == 3:
                return True
        
        if np.abs(np.trace(state)) == 3 or np.abs(np.trace(np.fliplr(state))) == 3:
            return True

        if legal == []:
            self.draw = True
            return True
        
        return False



    def action(self):
        legal, turn, state = self.get_obs()
        playing = self.p0 if turn == -1 else self.p1

        move = playing.move(legal, state)
        self.state[move[0]][move[1]] = turn
        legal, turn, state = self.get_obs()
        # print(self.render())
        if self.isTerminated() and not self.draw:
            return turn #the last move resulted in a win
        elif self.draw:
            return 2 #the game resulted in a draw

        self.turn = (-1) * turn
        return 0



    def play(self):
        i = 0
        while True:
            i = self.action()
            if i != 0:
                break
            self.p0.reward(0)
            self.p1.reward(0)

        match i:
            case -1:
                self.p0.reward(1)
                self.p1.reward(-1)
                self.p0_wins+=1

                # print("\n\nPlayer0 won!")
                

            case 1:
                self.p0.reward(-1)
                self.p1.reward(1)
                self.p1_wins+=1
                if self.p1.act_space.get(np.array2string(self.state)) is not None: print(self.p1.act_space[np.array2string(self.state)])

                # print("\n\nPlayer1 won!")
                
            case other:
                self.p0.reward(0)
                self.p1.reward(0)
                self.draws +=1

                # print("\n\nThe game resulted in a draw.")
                
        # print(self.render())
        self.reset()



    def render(self):
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
