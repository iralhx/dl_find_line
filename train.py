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

def train_logger(num):
    logger = logging.getLogger(__name__)
    #设置打印的级别，一共有6个级别，从低到高分别为：
    #NOTEST、DEBUG、INFO、WARNING、ERROR、CRITICAL。
    #setLevel设置的是最低打印的级别，低于该级别的将不会打印。
    logger.setLevel(level=logging.INFO)
    #打印到文件，并设置打印的文件名称和路径
    # file_log = logging.FileHandler('./run/{}/train.log'.format(num))
    #打印到终端
    print_log = logging.StreamHandler()
    #设置打印格式
    #%(asctime)表示当前时间，%(message)表示要打印的信息，用的时候会介绍。
    formatter = logging.Formatter('%(asctime)s     %(message)s')
    # file_log.setFormatter(formatter)
    print_log.setFormatter(formatter)

    # logger.addHandler(file_log)
    logger.addHandler(print_log)
    return logger


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
md = MyDataset('./dataset/train/',shuffle=True,lenght=20000)
net = ConModel(1)
net.cuda()
dl = DataLoader(md,batch_size=128)
log =train_logger(1)
num_epochs = 300
optimizer = torch.optim.Adadelta(net.parameters(), lr=0.001)
bset_loss=999
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
