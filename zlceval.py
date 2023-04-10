
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataser import *
import torch.backends.cudnn as cudnn
import cv2

def zlceval(net,evalset,error=0.05):
    # net.eval()
    count = 0
    error = 0.05
    for imgs, targets,_ in iter(evalset):
        imgs = imgs.cuda()
        imgs =imgs.reshape(1,1,256,256)
        k_b= net(imgs).cpu()
        k=float(k_b)

        if abs(targets-k)<error:
            # print('zlceval ,predict k : %f, label k : %f' % ( k , targets ))
            count+=1
    # net.train()
    return count/len(evalset)



