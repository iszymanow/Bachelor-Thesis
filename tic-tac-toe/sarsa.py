import numpy as np

class Sarsa:

    def __init__(self, alpha, eps, gamma):
        self.act_space = {} #the action value space Q
        self.alpha = alpha  #step size
        self.eps = eps      #exploration parameter
        self.gamma = gamma  #discounting rate

        self.curr_state = None  #S
        self.curr_action = None #A
        


    def move(self, state, legal):
        """
        the function which is responsible for choosing the move, given the current state of the game,
        as well as the list of legal moves. It returns a tuple (x,y) which indicates which place on
        the board the agent chooses
        """
       
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
            A_p = legal[index]
        else:
            A_p = max(self.act_space[a2s], key=self.act_space[a2s].get)

        return A_p            



    def initialize_sarsa(self, legal, S):
        """
        Function run at the beginning of the algorithm, it initializes the first state and chooses the first action.
        The return value is the chosen action A.
        """
        A = self.move(S, legal) #get A
        self.curr_state = S
        self.curr_action = A

        return A 



    def update_sarsa(self, S, A, R, S_p, A_p, terminal):
        """
        The function takes the key SARSA values, as well as termination indication and updates the action value space Q accordingly
        """
        # get string representations of S, S'
        a2s = np.array2string(S)
        a2ns = np.array2string(S_p)

        if not terminal:
            change = self.alpha * (R + self.gamma * self.act_space[a2ns][A_p] - self.act_space[a2s][A])
        else:
            # print(S,A,R)
            change = self.alpha * (R - self.act_space[a2s][A])
        self.act_space[a2s][A] += change
        self.curr_state = S_p
        self.curr_action = A_p


  
   
    def step_sarsa(self, R, S_p, legal, terminal):
        """
        Function which does one step of SARSA algorithm: it takes the most recent reward R, 
        the next state S_p as well as legal actions and termination indicator,
        chooses the next action and proceeds with the SARSA update.
        The return value is the next action A_p 
        """

        A_p = None
        if not terminal: 
            A_p = self.move(S_p, legal)
        self.update_sarsa(self.curr_state, self.curr_action, R, S_p, A_p, terminal)

        return A_p
