import numpy as np

class Sarsa:

    def __init__(self, sym, env, alpha, eps, gamma):
        self.act_space = {} #the action value space Q
        self.alpha = alpha  #step size
        self.eps = eps      #exploration parameter
        self.gamma = gamma  #discounting rate
        self.env = env      #the environment that the agent interacts with
        self.sym = sym      #the "symbol" of the player (used to determine if it's the player's turn to move)

        self.curr_state = None  #S
        self.curr_action = None #A
        self.reward = 0         #R
        self.next_state = None  #S'
        self.next_action = None #A'
        


    def move(self, state, legal):
        """
        the function which is responsible for choosing the move, given the current state of the game,
        as well as the list of legal moves. It returns a tuple (x,y) which indicates which place on
        the board the agent chooses
        """
        if self.env.isTerminated(state, legal): #if the game is already finished, just return a dummy output
            return (-1,-1)
        else:
            a2s = np.array2string(state) #convert the state into a string, so that it is hashable

            #initialize the entries in the action-val. space if state encountered for the 1st time
            if self.act_space.get(a2s) is None:
                self.act_space[a2s] = dict()
                for move in legal:
                    self.act_space[a2s][move] = 0

            #choose the move according to eps-greedy policy
            explore = np.random.random()
            if explore < self.eps:
                index = np.random.randint(len(legal))
                return legal[index]
            else:
                return max(self.act_space[a2s], key=self.act_space[a2s].get)
    


    def play(self):
        """
        The main functionality of the agent: function which plays the game (1 episode) and runs SARSA algorithm.

        """
        # reset the object's variables for new episode
        self.curr_state = None
        self.curr_action = None
        self.reward = 0
        self.next_state = None
        self.next_action = None
        
        with self.env.cv: # lock the environment, so that the other player doesn't change the state
            while self.env.turn != self.sym: # wait until it's the agent's turn
                self.env.cv.wait()

            legal, state = self.env.get_obs() #get S
            self.curr_state = np.copy(state)
            self.curr_action = self.move(self.curr_state, legal) #get A

        while True: # main SARSA loop
            with self.env.cv:
                self.reward = self.env.action(self.curr_action) #get R

                self.env.cv.notify() # wake the opponent up, as it's their turn now

                while self.env.turn != self.sym: # wait until the opponent makes their move
                    self.env.cv.wait()
           
                legal, state = self.env.get_obs() # get S'
                self.next_state = np.copy(state)
    
            self.next_action = self.move(self.next_state, legal) # get A'

            # get string representations of S, S'
            a2s = np.array2string(self.curr_state)
            a2ns = np.array2string(self.next_state)

            if self.next_action != (-1,-1): # S' not is terminal
                # SARSA update
                change = self.alpha * (self.reward + self.gamma * self.act_space[a2ns][self.next_action] - self.act_space[a2s][self.curr_action])
                self.act_space[a2s][self.curr_action] += change

                # S <- S', A <- A'
                self.curr_state = self.next_state
                self.curr_action = self.next_action
            else:
                with self.env.cv: 
                    if self.reward != 1: # last agent's move wasn't a direct win, but the game is now terminated => either it's a loss or a draw
                        self.reward = self.env.action((-1,-1)) # call action just to get the right reward; the state won't change
                    
                    self.env.cv.notify() # make sure the opponent can also terminate

                # SARSA update for case when S' is terminal [Q(S',*) = 0 for any terminal S']
                change = self.alpha * (self.reward - self.act_space[a2s][self.curr_action])
                self.act_space[a2s][self.curr_action] += change

                break # terminate, as the game is finished and all updates are done

        return 0