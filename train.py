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
from zlceval import *

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


loss = MSELoss()
# loss = nn.SmoothL1Loss(beta=0.05)
md = KpDataset('./dataset/train/',shuffle=True,lenght=1000)
evalDataset = KpDataset('./dataset/test/')
net = FullConModel()
net.cuda()
dl = DataLoader(md,batch_size=16)
accum_step=1
num_epochs = 100

# Adagrad Adam SparseAdam AdamW ASGD LBFGS RMSprop Rprop
# Adadelta
# optimizer = torch.optim.Adadelta( net.parameters() , lr=0.01)
optimizer = torch.optim.SGD( net.parameters() , lr=0.01)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

bset_ecval = 999
epoch_last_loss = 0
step =0
for epoch in range(1, num_epochs + 1):
    pbar = enumerate(dl)
    for i, (imgs, targets,_) in iter(pbar):
        step+=1
        imgs=imgs.cuda()
        targets=targets.cuda()
        net_out=net(imgs)
        l = loss(net_out, targets)
        epoch_last_loss = l
        l = l/accum_step
        l.backward()
        if step % accum_step ==0:
            optimizer.step()
            optimizer.zero_grad()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)  # clip gradient
    scheduler.step()
    ecval = zlceval(net,evalDataset,loss)
    if ecval<bset_ecval:
        bset_ecval=ecval
        torch.save(net,"net_best.pt")
    print('epoch %d, bset_ecval: %f' % (epoch, bset_ecval))
    print('epoch %d, loss: %f' % (epoch, epoch_last_loss))

torch.save(net,"net.pt")
