import torch
import numpy as np
import multiprocessing, concurrent.futures

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
        self.nonprogress = 0
        self.positions = {tuple(self.state.tolist()): 1}

        self.draw = False
        self.done = False

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


    def get_obs(self, needMask=True):
        state = self.state.clone()
        repetitions = self.positions[tuple(state.tolist())]
        non_p = self.nonprogress
        if self.turn == 1:
            state = state.flip(0)
        mask = self.get_mask(self.turn)
        done = self.isTerminated(mask) if not self.done else True

        return state, mask, done, repetitions, non_p


    def isTerminated(self, mask):
        
        # the player whose turn is now is blocked/have no pieces left, they lose
        if not mask.any():
            self.done = True
            if self.turn == -1:
                self.p1_wins += 1
            else:
                self.p0_wins += 1
            return self.done
        # 3th repetition of the position results in a draw
        key = tuple(self.state.tolist())
        if (self.positions.get(key) is not None) and self.positions[key] >= 3:
            self.draw = True
            # print('repeated position')
            self.draws += 1
            self.done=True
        # none of the players capturing any piece during their last 40 moves results in a draw
        elif self.nonprogress >= 40:
            self.draw = True
            self.draws += 1
            self.done=True
        
        return self.done


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

        board = self.state.view([8,8])
        # print(self.turn, board)
        
        # flip the board in case it's white's move
        if player == 1:
            board = self.flip_board(board)

        # executor = concurrent.futures.ProcessPoolExecutor(8)
        # futures = [executor.submit(self.checkOptions,board,player,i,j) for i in range(8) for j in range(8)]
        # concurrent.futures.wait(futures)
        # mask = torch.stack([future.result() for future in futures])

        if self.captureFlag != -1:
            mask = torch.zeros([64,8], dtype=torch.bool)
            mask[self.captureFlag] = self.checkOptions(board, player, self.captureFlag//8, self.captureFlag%8)
        else:
            mask = torch.stack([self.checkOptions(board, player, i, j) for i in range(8) for j in range(8)])

        if mask[:,1::2].any():
            mask[:,::2] = False

        return mask
    

    def checkOptions(self, board, player, i, j):
        mask = torch.zeros([8], dtype=torch.bool)
        if (i%2 + j) % 2 != 0:
            return mask
        if board[i][j] * player > 0:
                    # forward moves - check if they're even possible
                    if i < 7:
                        # left forward - check if moves to the left are even possible
                        if j > 0:
                            # check if the square is empty
                            if board[i+1][j-1] == 0:
                                mask[0] = True
                            # left capture - check if dimensions allow and whether capture is possible
                            elif i < 6 and j > 1 and board[i+1][j-1] * player < 0 and board[i+2][j-2] == 0:
                                mask[1] = True

                        # right forward - analogous as left moves
                        if j < 7:
                            if board[i+1][j+1] == 0:
                                mask[2] = True
                            # right capture
                            elif i < 6 and j < 6 and board[i+1][j+1] * player < 0 and board[i+2][j+2] == 0:
                                mask[3] = True

                    
                    # backward moves (only for kings)
                    if board[i][j] * player == 2:
                        if i > 0:
                            # left backward
                            if j > 0:
                                if board[i-1][j-1] == 0:
                                    mask[4] = True
                                # left_back capture
                                elif i > 1 and j > 1 and board[i-1][j-1] * player < 0 and board[i-2][j-2] == 0:
                                    mask[5] = True

                            # right backward
                            if j < 7:
                                if board[i-1][j+1] == 0:
                                    mask[6] = True
                                # right_back capture
                                elif i > 1 and j < 6 and board[i-1][j+1] * player < 0 and board[i-2][j+2] == 0:
                                    mask[7] = True


        
        return mask


    def decodeAction(self, action):
        action = action.squeeze()
        move = action % 8
        start = (action - move)//8
        numSquares = torch.tensor([7,14,9,18,-9,-18,-7,-14], device=device, dtype=torch.int8)
        # if self.turn == 1:
        #     numSquares *= (-1)
        #     start = 63 - start
        return (start, start+numSquares[move])


    def flip_board(self, board):
        return board.flip(1).flip(0)
                

    def step_env(self, action):
        
        reward0, reward1 = 0,0
        self.captureFlag = -1
        state = self.state
        if self.turn == 1:
            state = state.flip(0)

        done = self.done
        if done:
            if not self.draw:
                if self.turn == -1:
                    reward0, reward1 = -1,1
                else:
                    reward0, reward1 = 1,-1
        else:
            (start, end) = self.decodeAction(action)
            promotion = False
            capture = False

            if abs(start-end) == 14 or abs(start-end) == 18:
                capture = True
                # remove the captured piece from the board
                state[min(start, end) + abs(start-end)//2] = 0

            # check if progress was made (either a man move or a capture)
            if abs(state[start]) == 2 and (not capture):
                self.nonprogress += 0.5 # add 1/2 as we need 40 non-progress moves made by EACH player, not only by one 
            else:
                self.nonprogress = 0
                

            #the piece moves to the new square
            state[end] = state[start]
            state[start] = 0

            # check if there was a promotion to a king
            if abs(state[end]) == 1 and end >= 55:
                promotion = True
                state[end] *= 2

            capture_mask = self.checkOptions(state.view(8,8), self.turn, end//8, end%8)
            if self.turn == 1:
                state = state.flip(0)

            key = tuple(state.tolist())
            if self.positions.get(key) is None:
                self.positions[key] = 1
            else:
                self.positions[key] += 1

            self.state = state


            # if a player captured a piece, didn't promote to the king and still has captures available,
            # make sure that the same player gets another move
            if (not promotion) and capture and (capture_mask[1] or capture_mask[3] or capture_mask[5] or capture_mask[7]):
                # print('must capture again')
                self.turn *= (-1)
                self.captureFlag = end

        self.turn *= (-1)

        return reward0,reward1

            
    def render(self, orient=-1, manualAid=False):
        # reshape the state to a board form and orient it s.t. the black pieces are at the bottom
        state, mask, done, rep, nonp = self.get_obs()
        
        state = state.view([8,8])
        if orient == -1:
          state = self.flip_board(state)
        state = state.flip(0)
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
            aid = board + "number of repetitions of the position: " + str(rep) + "\nnumber of non-progress moves: " + str(nonp * 2) + "\nSquares with pieces that you can move:\n+---+---+---+---+---+---+---+---+\n"
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