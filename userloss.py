import torch.nn as nn

def uloss(output,terget):
    
    a=output.gt(terget.reshape(8,1))
    output[a]=0
    return nn.MSELoss(output,terget)

