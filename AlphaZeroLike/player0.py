import numpy as np
import torch

class RandomPlayer:
    """
    class that implements a dummy player which just plays random moves
    """
    def step_agent(self, mask):
        if not mask.any():
            return None
        else:
            legal = torch.tensor(range(len(mask)), dtype=torch.int64)[mask]
            idx = torch.randint(high=len(legal),size=(1,))
            return legal[idx].view(1,1)
      
class DummyPlayer:
    """
    Implementation of the player who always chooses the first available move (by iterating row by row)
    """
    def step_agent(self, mask, isTerminated):
        if isTerminated:
            return None
        else:
            for i in range(len(mask)):
                if mask[i]:
                    return torch.tensor([i])
                    break








