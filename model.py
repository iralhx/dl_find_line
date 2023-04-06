import torch
import torch.nn as nn
import math


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
        self.s = nn.Sigmoid()
        self.reset_parameters()
 
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, input):
        output = torch.zeros_like(input,dtype=float).cuda()
        output[:,0]  =self.s(input[:,0])
        output[:,1]  = self.mind_k + input[:,1] * self.weight
        return output


class model(nn.Module):
   def __init__(self, num_class):
       super(model,self).__init__()
       C = num_class
       self.conv_layer1=nn.Sequential(
           nn.Conv2d(in_channels=1,out_channels=2,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(2),
           nn.ReLU()
      )#8*128*128
       self.a1=SpatialAttention()
       self.conv_layer2=nn.Sequential(
           nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=3//2),
           nn.BatchNorm2d(16),
           nn.ReLU()
      )#16*128*128
       self.conv_layer3=nn.Sequential(
           nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=3//2),
           nn.BatchNorm2d(32),
           nn.ReLU()
      )#32*128*128
       self.conv_layer4=nn.Sequential(
           nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=3//2),
           nn.BatchNorm2d(64),
           nn.ReLU()
      )#64*128*128
       self.conv_layer5=nn.Sequential(
           nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(64),
           nn.ReLU()
      )#64*64*64
       self.conv_layer6=nn.Sequential(
           nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(32),
           nn.ReLU()
      )#32*32*32
       self.conv_layer7=nn.Sequential(
           nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(32),
           nn.ReLU()
      )#32*16*16
       self.conv_layer8=nn.Sequential(
           nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(32),
           nn.ReLU()
      )#32*8*8
       self.flatten = Flatten()
       self.conn_layer1 = nn.Sequential(
           nn.Linear(in_features=32*8*8,out_features=1024),
           nn.Dropout(0.2),
           nn.Sigmoid())
       self.conn_layer2 = nn.Sequential(nn.Linear(in_features=1024,out_features=512),
           nn.Dropout(0.2),
           nn.ReLU())
       self.conn_layer3 = nn.Sequential(nn.Linear(in_features=512,out_features=256),
           nn.Dropout(0.2),
           nn.ReLU())
       self.conn_layer4 = nn.Sequential(nn.Linear(in_features=256,out_features=128),
           nn.Dropout(0.2),
           nn.ReLU())
       self.conn_layer5 = nn.Sequential(nn.Linear(in_features=128,out_features=64),
           nn.Dropout(0.2),
           nn.ReLU())
       self.conn_layer6 = nn.Sequential(nn.Linear(in_features=64,out_features=32),
           nn.Dropout(0.2),
           nn.ReLU())
       self.conn_layer7 = nn.Sequential(nn.Linear(in_features=32,out_features=1))
    #    self.kk = kk(0.6,1)
       self._initialize_weights()
       
   def forward(self,input):
       output = self.conv_layer1(input)
       output = self.a1(output)
       output = self.conv_layer2(output)
       output = self.conv_layer3(output)
       output = self.conv_layer4(output)
       output = self.conv_layer5(output)
       output = self.conv_layer6(output)
       output = self.conv_layer7(output)
       output = self.conv_layer8(output)
       output = self.flatten(output)
       output = self.conn_layer1(output)
       output = self.conn_layer2(output)
       output = self.conn_layer3(output)
       output = self.conn_layer4(output)
       output = self.conn_layer5(output)
       output = self.conn_layer6(output)
       output = self.conn_layer7(output)
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
               m.bias.data.zero_()

