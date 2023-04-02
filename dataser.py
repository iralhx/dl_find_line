import os
import torch
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MyDataset(data.Dataset):
    def __init__(self,data_folder):
        self.data_folder = data_folder
        img_path=f'{data_folder}/images'
        label_path=f'{data_folder}/labels'
        self.h=256
        self.w=256
        self.filenames = []
        self.labels = []

        imgs = os.listdir(img_path)
        for single_img_path in imgs:
            label_img_paths = os.path.join(img_path, single_img_path)
            txtpath = f'{single_img_path.split(".")[0]}.txt'
            label_label_path=os.path.join(label_path, txtpath)
            if os.path.exists(label_label_path):
                self.filenames.append(label_img_paths)
                with open(label_label_path, 'r', encoding='utf-8') as f:
                        label = f.readline().split()
                        k=float(label[0])
                        self.labels.append(k)
        # self.labels.sort()
        # x = np.arange(len(self.labels))
        # plt.bar(x,self.labels)
        # plt.savefig(f'data.jpg')

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
            transforms.RandomCrop((self.h, self.w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ]
        return transforms.Compose(transform_train_list)(data)