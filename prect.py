import torch
import numpy as np
from dataser import *
import cv2
import time
from unit import setup_seed
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

setup_seed(3407)
path='net.pt'
net = torch.load(path).cuda()
net.eval()
for param in net.parameters():
    param.requires_grad = False
md = KpDataset('./dataset/eval/',shuffle=False,lenght=100)
i=1
base_path='run'
b=0
ok=[]
ok_k=[]
ng=[]
confi=0.87
confi_result=0.7
for imgs, _ ,path,l_k in iter(md):
    imgs = imgs.cuda()
    imgs =imgs.reshape(1,1,256,256)
    start=time.time()
    result= net(imgs).cpu()
    label_img = (result >= confi)
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
    spand=time.time() - start
    img = cv2.imread(path)
    cv2.line(img, (int(b),0), (256,int((256-b)/k)), (0, 255, 0), 1)

    write_path=f"{base_path}/{i}.jpg"
    cv2.imwrite(write_path,img)
    
    save_result = result.reshape(256, 256).numpy()
    # save_result[label_img>confi_result]=1
    # save_result[label_img<=confi_result]=0
    write_result_path=f"{base_path}/{i}_result.jpg"
    cv2.imwrite(write_result_path,save_result*255)
    
    logging.info('index %d, label_k: %f, result_k: %f ,time %f', i, l_k , k ,spand)
    i=i+1

