import torch.nn as nn
import math


class Flatten(nn.Module):
   def __init__(self):
       super(Flatten,self).__init__()
   def forward(self,x):
       return x.view(x.size(0),-1)
   
class model(nn.Module):
   def __init__(self, num_class):
       super(model,self).__init__()
       C = num_class
       self.conv_layer1=nn.Sequential(
           nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1,padding=7//2),
           nn.BatchNorm2d(64),
           nn.LeakyReLU(0.1),
           nn.MaxPool2d(kernel_size=2,stride=2)
      )
       self.conv_layer2=nn.Sequential(
           nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=3//2),
           nn.BatchNorm2d(192),
           nn.LeakyReLU(0.1),
           nn.MaxPool2d(kernel_size=2,stride=2)
      )
       #为了简便，这里省去了很多层
       self.flatten = Flatten()
       self.conn_layer1 = nn.Sequential(
           nn.Linear(in_features=7*7*1024,out_features=4096),
           nn.Dropout(0.5),nn.LeakyReLU(0.1))
       self.conn_layer2 = nn.Sequential(nn.Linear(in_features=4096,out_features=7*7*(2*5 + C)))
       
       self._initialize_weights()
       
   def forward(self,input):
       conv_layer1 = self.conv_layer1(input)
       conv_layer2 = self.conv_layer2(conv_layer1)
       flatten = self.flatten(conv_layer2)
       conn_layer1 = self.conn_layer1(flatten)
       output = self.conn_layer2(conn_layer1)
       return output
   
   def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
               m.weight.data.normal_(0, math.sqrt(2. / n))
               if m.bias is not None:
                   m.bias.data.zero_()
           elif isinstance(m, nn.BatchNorm2d):
               m.weight.data.fill_(1)
               m.bias.data.zero_()
           elif isinstance(m, nn.Linear):
               m.weight.data.normal_(0, 0.01)
               m.bias.data.zero_()