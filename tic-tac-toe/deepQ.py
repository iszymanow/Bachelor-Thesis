import torch
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class deepQNet(torch.nn.Module):

    def __init__(self):
        super(deepQNet, self).__init__()

        self.linear1 = torch.nn.Linear(9, 100)
        # self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100,9)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x



    
class DQN:

    def __init__(self, alpha, gamma, eps):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.act_space = {}
        self.curr_state = None
        self.curr_action= None
        self.episodes = 0

        self.behavior_net = deepQNet()
        self.target_net = deepQNet()

        self.optimizer = torch.optim.AdamW(self.behavior_net.parameters(), lr=1e-4, amsgrad=False)

    def move(self, state, legal, mask):
        if legal == []:
            return None
        A_p = None
        explore = np.random.random()
        if explore < self.eps:
            index = np.random.randint(len(legal))
            A_p = legal[index]
        else:
            shp = tuple(state.size())
            input = torch.reshape(state, (-1,))
            with torch.no_grad():
                output = self.behavior_net.forward(input)

            out = np.argmax(np.ma.array(output.detach().numpy(), mask=mask),axis=0)
            A_p = np.unravel_index(out, shp)

        return A_p
        

    def update_agent(self, S, A, R, S_p, terminal):
        if not terminal:

            target_Q_S_prime = torch.max(self.target_net.forward(S_p))
            target = self.alpha(R + self.gamma*target_Q_S_prime)
        else:
            target = R
        
        action_index = np.unravel_index(A, tuple(torch.size(S)))
        pred = self.behavior_net.forward(S)[action_index]

        crit =torch.nn.SmoothL1Loss()
        loss = crit(pred * np.sqrt(self.alpha), target * np.sqrt(self.alpha))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.curr_state = S_p

        

        if self.episodes % 50 == 0:
            self.target_net.load_state_dict(self.behavior_net.state_dict())
     
 

    def step_agent(self, A, R, S_p, legal, mask, terminal):
        A_p = None
        if not terminal: 
            A_p = self.move(S_p, legal, mask)
        else:
            self.episodes += 1
            
            
        if self.curr_state != None:
            self.update_agent(self.curr_state, A, R, S_p, terminal)

        return A_p

