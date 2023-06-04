import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, model, reg):
        super(CustomLoss, self).__init__()
        self.model = model
        self.reg = reg
    
    def forward(self, output_v, output_p, target_v, target_p):
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        l2 = 0
        for param in self.model.parameters:
            l2 += torch.sum(param.data**2)

        loss = mse(output_v, target_v) + ce(output_p, target_p) + self.reg * l2

        return loss
    

