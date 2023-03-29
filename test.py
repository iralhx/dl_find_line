from dataser import MyDataset
path='dataset'
a=MyDataset(path)
import torch
print(torch.backends.cudnn.version())

