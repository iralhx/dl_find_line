import torch
import torch.nn as nn
import math
from transformersmodel import *


class Flatten(nn.Module):
   def __init__(self):
       super(Flatten,self).__init__()
   def forward(self,x):
       return x.view(x.size(0),-1)
   
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output
    

class kk(nn.Module):
    def __init__(self, max_k,min_k):            
        super(kk, self).__init__()
        self.max_k = max_k
        self.min_k = min_k
        self.mind_k =(max_k+min_k)/2
        self.weight = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()
 
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, input):
        output = torch.zeros_like(input,dtype=float).cuda()
        output = self.mind_k + input * self.weight
        return output


class ConModel(nn.Module):
   def __init__(self, num_class):
       super(ConModel,self).__init__()
       C = num_class
       channels=4
       self.conv_layer1=nn.Sequential(
           nn.Conv2d(in_channels=1,out_channels=channels,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(channels),
           nn.ReLU()
      )#8*128*128
       self.a1=SpatialAttention()
       self.conv_layer2=nn.Sequential(
           nn.Conv2d(in_channels=1,out_channels=channels,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(channels),
           nn.ReLU()
      )#8*64*64
       self.a2=SpatialAttention()
       self.conv_layer3=nn.Sequential(
           nn.Conv2d(in_channels=1,out_channels=channels,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(channels),
           nn.ReLU()
      )#8*32*32
       self.a3=SpatialAttention()
       self.conv_layer4=nn.Sequential(
           nn.Conv2d(in_channels=1,out_channels=channels,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(channels),
           nn.ReLU()
      )#8*16*16
       self.a4=SpatialAttention()
       self.conv_layer5=nn.Sequential(
           nn.Conv2d(in_channels=1,out_channels=channels,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(channels),
           nn.ReLU()
      )#8*8*8
       self.a5=SpatialAttention()
       self.conv_layer6=nn.Sequential(
           nn.Conv2d(in_channels=1,out_channels=channels,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(channels),
           nn.ReLU()
      )#4*4*4
       self.flatten = Flatten()
       self.tr1=TransformerLayer(64,64)
       self.conn_layer1 = nn.Sequential(nn.Linear(in_features=64,out_features=16),
           nn.Dropout(0.2),
           nn.ReLU())
       self.conn_layer2 = nn.Sequential(nn.Linear(in_features=16,out_features=1))
    #    self.kk = kk(0.6,1)
       self._initialize_weights()
       
   def forward(self,input):
       output = self.conv_layer1(input)
       output = self.a1(output)
       output = self.conv_layer2(output)
       output = self.a2(output)
       output = self.conv_layer3(output)
       output = self.a3(output)
       output = self.conv_layer4(output)
       output = self.a4(output)
       output = self.conv_layer5(output)
       output = self.a5(output)
       output = self.conv_layer6(output)
       output = self.flatten(output)
       output = self.tr1(output)
       output = self.conn_layer1(output)
       output = self.conn_layer2(output)
    #    output = self.kk(output)
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
               if m.bias != None:
                    m.bias.data.zero_() 

