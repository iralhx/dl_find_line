import logging
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from dataser import *
from model import *
from userloss import *
from torch.optim.lr_scheduler import StepLR

#基本配置
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
#使用方法
#在train文件下调用下面这行命令，参数随意设置，
#只要这个参数数值一样，每次生成的顺序也就一样
setup_seed(3407)


#loss = MSELoss(100)
loss = nn.SmoothL1Loss(beta=0.05)
md = MyDataset('./dataset/train/',shuffle=True,lenght=1000)
net = ConModel(1)
net.cuda()
dl = DataLoader(md,batch_size=128)
num_epochs = 100
# Adagrad Adadelta Adam SparseAdam AdamW ASGD LBFGS RMSprop
# RMSprop Rprop
optimizer = torch.optim.RMSprop( net.parameters() , lr=0.005)
bset_loss = 999
for epoch in range(1, num_epochs + 1):
    pbar = enumerate(dl)
    for i, (imgs, targets,_) in iter(pbar):
        imgs=imgs.cuda()
        targets=targets.cuda()
        net_out=net(imgs).to(torch.float64)
        l = loss(net_out, targets.to(torch.float64))
        if bset_loss>l:
            bset_loss=l
            torch.save(net,"net_best.pt")
            print('epoch %d, best_loss: %f' % (epoch, l))
        optimizer.zero_grad()
        l.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)  # clip gradients
        optimizer.step()
        print('epoch %d, loss: %f' % (epoch, l))
torch.save(net,"net.pt")
print('finel bset loss: %f' % (bset_loss))
