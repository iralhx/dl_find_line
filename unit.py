import numpy as np
import random
import torch.backends.cudnn as cudnn
import torch
import torch.backends.cudnn as cudnn

#基本配置
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True