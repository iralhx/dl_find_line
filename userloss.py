import torch.nn as nn

def uloss(output,terget):
    a=output.gt(terget.reshape(8,1))
    output[a]=0
    return nn.MSELoss(output,terget)

class MSELoss(nn.Module):
    def __init__(self, weight=1):
        super(MSELoss, self).__init__()
        self.weight = weight
    
    def forward(self, input, target):
        loss = (self.weight * (input - target) ** 2).mean()
        return loss
