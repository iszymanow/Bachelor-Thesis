import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, numChannels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(numChannels, 256, [3,3], 1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        return x
    
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(6, 256, [3,3], 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, [3,3], 1)
        self.bn2 = nn.BatchNorm2d(256)


    def forward(self, x):
        x = x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        x = F.relu(x)

        return x
    
class Heads(nn.Module):
    def __init__(self, numActions):
        super(Heads, self).__init__()
        # value
        self.conv1 = nn.Conv2d(256,1,[1,1],1)
        self.bn1 = nn.BatchNorm2d(1)
        self.lin1 = nn.Linear(6*256, 256)
        self.lin2 = nn.Linear(256,1)

        # policy
        self.conv2 = nn.Conv2d(256,2,[1,1],1)
        self.bn2 = nn.BatchNorm2d(2)
        self.lin3 = nn.Linear(6 * 2 * 256, numActions)

    def forward(self, x):
        v = self.conv1(x)
        v = self.bn1(v)
        v = F.relu(v)
        v = v.flatten()
        v = self.lin1(v)
        v = F.relu(v)
        v = self.lin2(v)
        v = F.tanh(v)

        p = self.conv2(x)
        p = self.bn2(p)
        p = F.relu(p)
        p = p.flatten()
        p = self.lin3(p)
        p = F.softmax(p)

        return p, v


class CheckersNN():
    def __init__(self, num_obs, num_actions, numResBlocks):
        super(CheckersNN, self).__init__()
        self.convBlock = ConvBlock(num_obs)
        for res in range(1,numResBlocks+1):
            setattr(self, "resBlock" + str(res), ResBlock())
        self.heads = Heads(num_actions)

        self.numResBlocks = numResBlocks

    def forward(self, x):
        x = self.convBlock(x)
        for res in range(1,self.numResBlocks + 1):
            x = getattr(self, "resBlock" + str(res))(x)
        p,v = self.heads(x)

        return p,v

    def encodeState(self, board, numRepetitions, nonProgress):
        state = board.view(8,8)

        encoded = torch.zeros([6,8,8])

        encoded[0,:,:][state == -1] = 1
        encoded[1,:,:][state == -2] = 1
        encoded[2,:,:][state == 1] = 1
        encoded[3,:,:][state == 2] = 1
        encoded[4,:,:] = numRepetitions
        encoded[5,:,:] = nonProgress

        return encoded
    
