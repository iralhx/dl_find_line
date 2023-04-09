import logging
import time
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


# loss = MSELoss(10)
loss = nn.SmoothL1Loss(beta=0.05)
md = MyDataset('./dataset_small/train/',shuffle=True,lenght=100)
net = ResModel1()
net.cuda()
dl = DataLoader(md,batch_size=2)
accum_step=8
num_epochs = 30


# Adagrad Adam SparseAdam AdamW ASGD LBFGS RMSprop Rprop
# Adadelta
optimizer = torch.optim.Adadelta( net.parameters() , lr=0.01)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

bset_loss = 999
step =0
all_loss = 0
s_t = time.time()
for epoch in range(1, num_epochs + 1):
    pbar = enumerate(dl)
    for i, (imgs, targets,_) in iter(pbar):
        step+=1
        imgs=imgs.cuda()
        targets=targets.cuda()
        net_out=net(imgs).to(torch.float64)
        l = loss(net_out, targets.to(torch.float64))
        l = l/accum_step
        all_loss+=l
        l.backward()
        if step % accum_step ==0:
            optimizer.step()
            optimizer.zero_grad()
            start_t1 = time.time() - s_t
            s_t = time.time()
            print('epoch %d, loss: %f , one update: %f s'% (epoch, all_loss,start_t1))
            if bset_loss>all_loss:
                bset_loss=all_loss
                torch.save(net,"resnet_small_best.pt")
                print('epoch %d, best_loss: %f' % (epoch, all_loss))
            all_loss= 0
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)  # clip gradient
    scheduler.step()
torch.save(net,"resnet_small.pt")
print('finel bset loss: %f' % (bset_loss))
