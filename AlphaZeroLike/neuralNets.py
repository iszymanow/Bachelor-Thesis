import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, numChannels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=256, kernel_size=3, stride=1,padding=1)
        torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=0.1) 
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        return x
    
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
        torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=0.1) 
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding=1)
        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=0.1) 
        self.bn2 = nn.BatchNorm2d(256)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # print(out.size())
        out = out + x
        out = F.relu(out)

        return out
    
class Heads(nn.Module):
    def __init__(self, numActions):
        super(Heads, self).__init__()
        # value
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1,padding=0)
        torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=0.1)
        self.bn1 = nn.BatchNorm2d(1)
        self.lin1 = nn.Linear(8*8, 256)
        torch.nn.init.normal_(self.lin1.weight, mean=0.0, std=0.1) 
        self.lin2 = nn.Linear(256,1)
        torch.nn.init.normal_(self.lin2.weight, mean=0.0, std=0.1) 


        # policy
        self.conv2 =  nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1,padding=0)
        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=0.1) 
        self.bn2 = nn.BatchNorm2d(2)
        self.lin3 = nn.Linear(2*8*8, numActions)
        torch.nn.init.normal_(self.lin3.weight, mean=0.0, std=0.1) 


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
        p = F.softmax(p, dim=0)
        
        return p, v


class CheckersNN(nn.Module):
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
    
