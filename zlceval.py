
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataser import *
import torch.backends.cudnn as cudnn
import cv2

def zlceval(net,evalset ,loss):
    net.eval()
    all_loss=0
    for imgs, targets,_,_ in iter(evalset):
        imgs = imgs.cuda()
        imgs =imgs.reshape(1,1,256,256)
        k= net(imgs).cpu()
        all_loss+=loss(imgs.cpu(),k)
    net.train()
    return all_loss/len(evalset)



