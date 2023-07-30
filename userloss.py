import torch.nn as nn
import torch
def uloss(output,terget):
    a=output.gt(terget.reshape(8,1))
    output[a]=0
    return nn.MSELoss(output,terget)

class MSELoss(nn.Module):
    def __init__(self, weight=1):
        super(MSELoss, self).__init__()
        self.weight = weight
    
    def forward(self, input, target,k):

        loss = 0
        for i in range(k.shape[0]):

            input_k_col = input[:, :, :, k[i]]
            target_k_col = target[:, :, k[i]]
            loss += torch.mean((input_k_col - target_k_col) ** 2)


        loss = loss * self.weight

        return loss
