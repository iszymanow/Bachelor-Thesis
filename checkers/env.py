import torch
import numpy as np

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'

class Env:
    """The environment implementation for standard checkers.
    
    State representation: a tensor representing an 8x8 board. Empty fields are filled with 0s,
    white uncrowned pieces are represented with 1s and crowned with 2s.
    Analogously, black uncrowned pieces are represented with -1s and crowned with -2s.

    Internally the board tensor is flattened

    Action representation: a tuple indicating which piece to move, as well as the end position of the piece.
    Internally it's (x,y), where x indicates the index of the (flattened) board where the moved piece is and
    y is the index of the square where the piece end up
    """

    def __init__(self) -> None:
        """
        initialize the game environment, along with results
        """
        self.turn=-1
        
        row0 = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.int8, device=device)
        row1 = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int8, device=device)
        rowBlank = torch.zeros(size=[8], dtype=torch.int8, device=device)

        self.state = torch.cat([-row0, -row1, -row0, rowBlank, rowBlank, row1, row0, row1])
        self.captureFlag = -1
        self.captures0, self.captures1 = 0,0
        self.man_moves0, self.man_moves1 = 0,0
        self.positions = {}

        self.draw = False

        self.p0_wins= 0
        self.p1_wins= 0
        self.draws = 0



    def reset(self):
        curr_p0_wins = self.p0_wins
        curr_p1_wins = self.p1_wins
        curr_draws = self.draws

        self.__init__()
        self.p0_wins = curr_p0_wins
        self.p1_wins = curr_p1_wins
        self.draws = curr_draws




    def get_obs(self):
        mask = self.get_mask(self.turn)
        done = self.isTerminated()

        return self.state.clone(), mask, done


    def isTerminated(self):
        
        # the player whose turn is now is blocked/have no pieces left, they lose
        if not self.get_mask(self.turn).any():
            # print('mask')
            return True
        # 3th repetition of the position results in a draw
        key = tuple(self.state.tolist())
        if (self.positions.get(key) is not None) and self.positions[key] >= 3:
            self.draw = True
            # print('repeated position')
            return True
        # none of the players capturing any piece during their last 40 moves results in a draw
        elif self.captures0 >= 40 and self.captures1 >= 40:
            self.draw = True
            # print('40 moves without capture')
            return True
        # none of the players moving a man piece during their last 40 moves results in a draw
        elif self.man_moves0 >= 40 and self.man_moves1 >= 40:
            # print('40 moves without man move')
            self.draw = True
            return True
        else:
            return False




    def get_mask(self, player):

        # possible actions:
        # 0 - left forward (man)
        # 1 - left capture (man)
        # 2 - right forward (man)
        # 3 - right capture (man)
        # 4 - left backward (king)
        # 5 - left_back capture (king)
        # 6 - right backward (king)
        # 7 - right_back capture (king)

        mask = torch.zeros([64, 8], dtype=torch.bool, device=device)
        board = self.state.reshape([8,8])
        forcedCapture = False

        # flip the board in case it's white's move
        if player == 1:
            board = self.flip_board(board)

        size = board.size()
        for i in range(size[0]):
            for j in range(size[1]):
                if board[i][j] * player > 0:
                    # forward moves - check if they're even possible
                    if i < 7:
                        # left forward - check if moves to the left are even possible
                        if j > 0:
                            # check if the square is empty
                            if board[i+1][j-1] == 0:
                                mask[8*i+j][0] = True
                            # left capture - check if dimensions allow and whether capture is possible
                            elif i < 6 and j > 1 and board[i+1][j-1] * player < 0 and board[i+2][j-2] == 0:
                                mask[8*i+j][1] = True
                                forcedCapture = True
                        # right forward - analogous as left moves
                        if j < 7:
                            if board[i+1][j+1] == 0:
                                mask[8*i+j][2] = True
                            # right capture
                            elif i < 6 and j < 6 and board[i+1][j+1] * player < 0 and board[i+2][j+2] == 0:
                                mask[8*i+j][3] = True
                                forcedCapture = True
                    
                    # backward moves (only for kings)
                    if board[i][j] * player == 2:
                        if i > 0:
                            # left backward
                            if j > 0:
                                if board[i-1][j-1] == 0:
                                    mask[8*i+j][4] = True
                                # left_back capture
                                elif i > 1 and j > 1 and board[i-1][j-1] * player < 0 and board[i-2][j-2] == 0:
                                    mask[8*i+j][5] = True
                                    forcedCapture = True
                            # right backward
                            if j < 7:
                                if board[i-1][j+1] == 0:
                                    mask[8*i+j][6] = True
                                # right_back capture
                                elif i > 1 and j < 6 and board[i-1][j+1] * player < 0 and board[i-2][j+2] == 0:
                                    mask[8*i+j][7] = True
                                    forcedCapture = True
                    
                    # mask out non-capturing moves if at least one capture move is available (CAPTURES ARE FORCED)
                    if forcedCapture:
                        mask[:,0] = False
                        mask[:,2] = False
                        mask[:,4] = False
                        mask[:,6] = False


        # if it's white's move, flip the mask "upside down", as the fields for them were considered in reverse order
        # (due to flipping the board at the beginning)
        if player == 1:
            mask = mask.flip(0) 
        # print(mask)
        return mask
    
    def decodeAction(self, action):
        action = action.squeeze()
        move = action % 8
        start = (action - move)//8
        numSquares = torch.tensor([7,14,9,18,-9,-18,-7,-14], device=device, dtype=torch.int8)
        if self.turn == 1:
            numSquares *= (-1)
        return (start, start+numSquares[move])
       


    def flip_board(self, board):
        return board.flip(1).flip(0)
                

    def step_env(self, action):
        
        reward = 0
        self.captureFlag = -1
        state, mask, done = self.get_obs()

        if done:
            if self.draw:
                reward = 0
            else:
                reward = -1
        else:
            (start, end) = self.decodeAction(action)
            promotion = False
            capture = False
            #the piece moves to the new square
            # self.state[start], self.state[end] = self.state[end], self.state[start]
            if abs(self.state[start]) == 2:
                if self.turn == -1:
                    self.man_moves0 += 1
                else:
                    self.man_moves1 += 1
            else:
                if self.turn == -1:
                    self.man_moves0 = 0
                else:
                    self.man_moves1 = 0
                
            self.state[end] = self.state[start]
            self.state[start] = 0

            
            if abs(self.state[end]) == 1 and ((self.turn == -1 and end >= 55) or (self.turn == 1 and end <= 7)):
                promotion = True
                self.state[end] *= 2

            # the chosen move was a capture
            if abs(start-end) == 14 or abs(start-end) == 18:
                capture = True
                # remove the captured piece from the board
                self.state[min(start, end) + abs(start-end)//2] = 0

                if self.turn == -1:
                    self.captures0 = 0
                else:
                    self.captures1 = 0
            else:
                if self.turn == -1:
                    self.captures0 += 1
                else:
                    self.captures1 += 1

            state, mask, done = self.get_obs()
            
            state = tuple(state.tolist())
            if self.positions.get(state) is None:
                self.positions[state] = 1
            else:
                self.positions[state] += 1

    
            self.turn *= (-1)
            # if a player captured a piece, didn't promote to the king and still has captures available,
            # make sure that the same player gets another move
            if (not promotion) and capture and (mask[end][1] or mask[end][3] or mask[end][5] or mask[end][7]):
                self.turn *= (-1)
                self.captureFlag = end
            else:
                if self.isTerminated():
                    if self.draw:
                        reward = 0
                        self.draws += 1
                    else:
                        reward = 1
                        if self.turn == -1:
                            self.p1_wins += 1
                        else:
                            self.p0_wins += 1

        return reward


            
    def render(self, orient=-1, manualAid=False):
        # reshape the state to a board form and orient it s.t. the black pieces are at the bottom
        state, mask, done = self.get_obs()
        
        state = state.reshape([8,8])
        if orient == -1:
            state = state.flip(0)
        else:
            state = self.flip_board(state).flip(0)
            mask = mask.flip(0)
        board = "\n+---+---+---+---+---+---+---+---+\n"
        for row in state:
            board += '|'
            for col in row:
                if col == -1:
                    board += ' o |'
                elif col == -2:
                    board += ' O |'
                elif col == 1:
                    board += ' x |'
                elif col == 2:
                    board += ' X |'
                else:
                    board += '   |'
            board += "\n+---+---+---+---+---+---+---+---+\n"

        if manualAid:
            actions = {}
            aid = board + "\nSquares with pieces that you can move:\n+---+---+---+---+---+---+---+---+\n"
            for i in range(state.size()[0]):
                i = 7 - i
                aid += '|'
                for j in range(state.size()[1]):
                    # j = 7 - j
                    if (i + j) % 2 == 0 and mask[8*i + j].any():
                        actions[8*i+j] = [index for index in range(len(mask[8*i + j])) if mask[8*i+j][index]]
                        if 8 * i + j < 10:
                            aid += ' ' + str(8*i+j) + ' |'
                        else:
                            aid += str((8*i+j)//10) + ' ' + str((8*i+j)%10) + '|'
                    else:
                        aid += '   |'
                aid += "\n+---+---+---+---+---+---+---+---+\n"

            board = aid

        print(board)
        if manualAid:
            print('Action legend:\n' +
                    '0 - left fw\n' +
                    '1 - left fw capture\n' +
                    '2 - right fw\n' +
                    '3 - right fw capture\n' +
                    '4 - left bw\n' +
                    '5 - left bw capture\n' +
                    '6 - right bw\n' +
                    '7 - right bw capture\n' +
                    'available actions:', actions)






# def main():
#     e = Env()
#     e.render()
#     print(e.get_mask(1))

# main()

