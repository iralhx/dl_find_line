
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataser import *
import torch.backends.cudnn as cudnn
import cv2
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
setup_seed(3407)
path='net_best.pt'
net = torch.load(path)
md = MyDataset('./dataset/test/')
net.cuda()
b=10
i=1
base_path='run'
for imgs, targets,path in iter(md):
    imgs = imgs.cuda()
    imgs =imgs.reshape(1,1,512,512)
    a = net(imgs)
    a=float(a.cpu())
    img = cv2.imread(path)
    path=f"{base_path}/{i}.jpg"
    i=i+1
    # 绘制点之间的连线
    cv2.line(img, (b,0), (512,int((512-b)/a)), (0, 255, 0), 2)
    cv2.imwrite(path,img)


    print('p %f, label: %f' % (a, targets))

