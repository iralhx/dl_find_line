import os
import torch
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class MyDataset(data.Dataset):
   def __init__(self,data_folder):
       self.data_folder = data_folder
       img_path=f'{data_folder}/images'
       label_path=f'{data_folder}/labels'
       self.h=512
       self.w=512
       self.filenames = []
       self.labels = []

       imgs = os.listdir(img_path)
       for single_img_path in imgs:
           label_img_paths = os.path.join(img_path, single_img_path)
           txtpath = f'{single_img_path.split(".")[0]}.txt'
           label_label_path=os.path.join(label_path, txtpath)
           if os.path.exists(label_label_path):
               self.filenames.append(label_img_paths)
               self.labels.append(label_label_path)
   def __getitem__(self, index):
       image = Image.open(self.filenames[index])
       label = self.labels[index]
       data = self.proprecess(image)
       return data, label

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