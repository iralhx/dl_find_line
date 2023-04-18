
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataser import *
import torch.backends.cudnn as cudnn
import cv2
import time

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
path='net.pt'
net = torch.load(path).cuda()
net.eval()
md = MyDataset('./dataset/eval/',shuffle=False,lenght=100)
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
    start=time.time()
    k_b= net(imgs).cpu()
    label_img = (k_b >= 0.7)
    label_img = label_img.reshape(256, 256).numpy().astype(np.uint8)
    row_idx, col_idx = np.where(label_img > 0)

    points=[]
    for index in range(len(row_idx)):
        points.append((row_idx[index],col_idx[index]))

    points=np.array(points)
    # 拟合直线
    output = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    # 计算直线起点和终点
    k = output[1] / output[0]
    b = output[3] - k * output[2]
    path=f"{base_path}/{i}.jpg"
    img = cv2.imread(path)
    cv2.line(img, (int(b),0), (256,int((256-b)/k)), (0, 255, 0), 1)

    # # 绘制点之间的连线
    # cv2.line(img, (int(b),0), (512,int((512-b)/k)), (0, 255, 0), 2)
    # label_img = (k_b >= 0.7)
    # label_img = label_img.reshape(256, 256).numpy().astype(np.uint8) 
    # # writeimg = np.zeros((256,256), dtype=np.float32)
    # # writeimg[label_img]=255
    # img = cv2.imread(path)
    # b, g, r = cv2.split(img)
    # b[label_img>0]=0
    # g[label_img>0]=255
    # r[label_img>0]=0
    # img = cv2.merge((b, g, r))
    # path=f"{base_path}/{i}.jpg"
    # # 绘制点之间的连线
    # cv2.line(img, (int(b),0), (512,int((512-b)/k)), (0, 255, 0), 2)
    cv2.imwrite(path,img)
    i=i+1

#     print('index : %d ,predict k : %f, label k : %f' % (i , k , targets ))
#     if abs( targets-k)>error:
#         ng.append(i)
#     else:
#         ok.append(i)
#         ok_k.append(targets)
#     i=i+1
# print ('avg once time : %d'%(all_time/len(md)))
# print ("OK: %d" % (len(ok)))
# print (ok)


# plt.clf()
# ok_k.sort()
# x = np.arange(len(ok_k))
# plt.bar(x,ok_k)
# plt.savefig(f'ok_k.jpg')


