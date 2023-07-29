import os
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random


class MyDataset(data.Dataset):
    def __init__(self,data_folder,draw_img=False,shuffle = False,lenght=99999999):
        self.data_folder = data_folder
        img_path=f'{data_folder}/images'
        label_path=f'{data_folder}/labels'
        self.h=256
        self.w=256
        self.filenames = []
        self.labels = []

        imgs = os.listdir(img_path)
        if shuffle:
            random.shuffle(imgs)
        sortLabels=[]
        for single_img_path in imgs:
            label_img_paths = os.path.join(img_path, single_img_path)
            txtpath = f'{single_img_path.split(".")[0]}.txt'
            label_label_path=os.path.join(label_path, txtpath)
            if os.path.exists(label_label_path):
                self.filenames.append(label_img_paths)
                with open(label_label_path, 'r', encoding='utf-8') as f:
                        label = f.readline().split()
                        k=float(label[0])
                        # b=float(label[1])
                        sortLabels.append(k)
                        # self.labels.append(torch.tensor([k,b]))
                        self.labels.append(k)
            if len(self.labels)>lenght:
                 break
        if draw_img:
            sortLabels.sort()
            x = np.arange(len(sortLabels))
            plt.bar(x,sortLabels)
            plt.savefig(f'data.jpg')

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        label = self.labels[index]
        data = self.proprecess(image)
        return data, label,self.filenames[index]

    def __len__(self):
        return len(self.filenames)

    def proprecess(self,data):
        transform_train_list = [
            transforms.Resize((self.h, self.w), interpolation=1),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        return transforms.Compose(transform_train_list)(data)


class KpDataset(data.Dataset):
    def __init__(self,data_folder,draw_img=False,shuffle = False,lenght=99999999):
        self.data_folder = data_folder
        img_path=f'{data_folder}/images'
        label_path=f'{data_folder}/labels'
        self.h=256
        self.w=256
        self.filenames = []
        self.labels = []

        imgs = os.listdir(img_path)
        if shuffle:
            random.shuffle(imgs)
        self.ks=[]
        for single_img_path in imgs:
            label_img_paths = os.path.join(img_path, single_img_path)
            txtpath = f'{single_img_path.split(".")[0]}.txt'
            label_label_path=os.path.join(label_path, txtpath)
            if os.path.exists(label_label_path):
                self.filenames.append(label_img_paths)
                with open(label_label_path, 'r', encoding='utf-8') as f:
                        label = f.readline().split()
                        k=float(label[0])
                        label_img = np.zeros((self.h,self.w), dtype=np.uint8)
                        self.ks.append(k)
                        cv2.line(label_img,(0,0),(self.h,int(self.h*k)),1)
                        # self.labels.append(torch.tensor([k,b]))
                        self.labels.append(label_img)
            if len(self.labels)>lenght:
                 break
    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        label = self.labels[index]
        data = self.proprecess(image)
        return data, label,self.filenames[index],self.ks[index]

    def __len__(self):
        return len(self.filenames)

    def proprecess(self,data):
        transform_train_list = [
            transforms.Resize((self.h, self.w), interpolation=1),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        return transforms.Compose(transform_train_list)(data)
    


class KpDatasetNew(data.Dataset):
    def __init__(self,data_folder,draw_img=False,shuffle = False,lenght=99999999):
        self.data_folder = data_folder
        img_path=f'{data_folder}/images'
        label_path=f'{data_folder}/labels'
        self.h=256
        self.w=256
        self.filenames = []
        self.labels = []

        imgs = os.listdir(img_path)
        if shuffle:
            random.shuffle(imgs)
        self.ks=[]
        for single_img_path in imgs:
            label_img_paths = os.path.join(img_path, single_img_path)
            txtpath = f'{single_img_path.split(".")[0]}.txt'
            label_label_path=os.path.join(label_path, txtpath)
            if os.path.exists(label_label_path):
                self.filenames.append(label_img_paths)
                with open(label_label_path, 'r', encoding='utf-8') as f:
                        label = f.readline().split()
                        k=int(label[0])
                        label_img = np.zeros((self.h,self.w), dtype=np.uint8)
                        self.ks.append(k)
                        cv2.line(label_img,(k,0),(k,self.h),1)
                        # self.labels.append(torch.tensor([k,b]))
                        self.labels.append(label_img)
            if len(self.labels)>lenght:
                 break
    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        label = self.labels[index]
        data = self.proprecess(image)
        return data, label,self.filenames[index],self.ks[index]

    def __len__(self):
        return len(self.filenames)

    def proprecess(self,data):
        transform_train_list = [
            transforms.Resize((self.h, self.w), interpolation=1),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        return transforms.Compose(transform_train_list)(data)
    

