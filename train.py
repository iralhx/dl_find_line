import logging
import numpy as np

import torch
def train_logger(num):
    logger = logging.getLogger(__name__)
    #设置打印的级别，一共有6个级别，从低到高分别为：
    #NOTEST、DEBUG、INFO、WARNING、ERROR、CRITICAL。
    #setLevel设置的是最低打印的级别，低于该级别的将不会打印。
    logger.setLevel(level=logging.INFO)
    #打印到文件，并设置打印的文件名称和路径
    file_log = logging.FileHandler('./run/{}/train.log'.format(num))
    #打印到终端
    print_log = logging.StreamHandler()
    #设置打印格式
    #%(asctime)表示当前时间，%(message)表示要打印的信息，用的时候会介绍。
    formatter = logging.Formatter('%(asctime)s     %(message)s')
    file_log.setFormatter(formatter)
    print_log.setFormatter(formatter)

    logger.addHandler(file_log)
    logger.addHandler(print_log)
    return logger


import random
import torch.backends.cudnn as cudnn
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
setup_seed(2022)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
def train():
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), labels)
        print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))