from dataser import MyDataset
path='dataset'
a=MyDataset(path)
data, label =a.__getitem__(1)


