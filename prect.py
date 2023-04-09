
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataser import *
import torch.backends.cudnn as cudnn
import cv2

path='resnet_small_best.pt'
net = torch.load(path)
net.eval()
md = MyDataset('./dataset/test/',shuffle=False,lenght= 100)
net.cuda()
i=1
base_path='run'
b=0
ok=[]
ok_k=[]
ng=[]
error = 0.05
for imgs, targets,path in iter(md):
    imgs = imgs.cuda()
    imgs =imgs.reshape(1,1,256,256)
    k_b= net(imgs).cpu()
    k=float(k_b)
    img = cv2.imread(path)
    path=f"{base_path}/{i}.jpg"
    # 绘制点之间的连线
    cv2.line(img, (int(b),0), (512,int((512-b)/k)), (0, 255, 0), 2)
    cv2.imwrite(path,img)

    print('index : %d ,predict k : %f, label k : %f' % (i , k , targets ))
    if abs( targets-k)>error:
        ng.append(i)
    else:
        ok.append(i)
        ok_k.append(targets)
    i=i+1
print ("OK: %d" % (len(ok)))
print (ok)

plt.clf()
ok_k.sort()
x = np.arange(len(ok_k))
plt.bar(x,ok_k)
plt.savefig(f'ok_k.jpg')


