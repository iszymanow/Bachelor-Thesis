import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, numChannels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=256, kernel_size=3, stride=1,padding=1)
        torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01) 
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # print(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
        torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01) 
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01) 
        self.bn2 = nn.BatchNorm2d(256)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = F.relu(out)

        return out
    
class Heads(nn.Module):
    def __init__(self, numActions, boardGrid):
        super(Heads, self).__init__()
        # value
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1,padding=0)
        torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        self.bn1 = nn.BatchNorm2d(1)
        self.lin1 = nn.Linear(boardGrid, 256)
        torch.nn.init.normal_(self.lin1.weight, mean=0.0, std=0.01) 
        self.lin2 = nn.Linear(256,1)
        torch.nn.init.normal_(self.lin2.weight, mean=0.0, std=0.01) 


        # policy
        self.conv2 =  nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1,padding=0)
        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01) 
        self.bn2 = nn.BatchNorm2d(2)
        self.lin3 = nn.Linear(2*boardGrid, numActions)
        torch.nn.init.normal_(self.lin3.weight, mean=0.0, std=0.01) 

        self.grid = boardGrid


    def forward(self, x):
        v = self.conv1(x)
        v = self.bn1(v)
        v = F.relu(v)
        v = v.view(-1,self.grid)
        v = self.lin1(v)
        v = F.relu(v)
        v = self.lin2(v)
        v = F.tanh(v)

        p = self.conv2(x)
        p = self.bn2(p)
        p = F.relu(p)
        p = p.view(-1,self.grid*2)
        p = self.lin3(p)
        p = F.softmax(p, dim=1)
        
        return p, v


class CheckersNN(nn.Module):
    def __init__(self,numResBlocks):
        super(CheckersNN, self).__init__()
        self.convBlock = ConvBlock(6)
        for res in range(1,numResBlocks+1):
            setattr(self, "resBlock" + str(res), ResBlock())
        self.heads = Heads(512, 8*8)

        self.numResBlocks = numResBlocks

    def forward(self, x):
        x = self.convBlock(x)
        for res in range(1,self.numResBlocks + 1):
            x = getattr(self, "resBlock" + str(res))(x)
        p,v = self.heads(x)

        return p,v
    
class TicTacToeNN(nn.Module):
    def __init__(self,numResBlocks):
        super(TicTacToeNN, self).__init__()
        self.convBlock = ConvBlock(2)
        for res in range(1,numResBlocks+1):
            setattr(self, "resBlock" + str(res), ResBlock())
        self.heads = Heads(9, 3*3)

        self.numResBlocks = numResBlocks

    def forward(self, x):
        x = self.convBlock(x)
        for res in range(1,self.numResBlocks + 1):
            x = getattr(self, "resBlock" + str(res))(x)
        p,v = self.heads(x)

        return p,v
    
