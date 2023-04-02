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

class model(nn.Module):
   def __init__(self, num_class):
       super(model,self).__init__()
       C = num_class
       self.conv_layer1=nn.Sequential(
           nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(1),
           nn.ReLU()
      )#1*128*128
       self.a1=SpatialAttention()
       self.conv_layer2=nn.Sequential(
           nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=2,padding=3//2),
           nn.BatchNorm2d(1),
           nn.ReLU()
      )#1*64*64
       self.a2=SpatialAttention()
       self.flatten = Flatten()
       self.conn_layer1 = nn.Sequential(
           nn.Linear(in_features=64*64,out_features=1024),
           nn.Dropout(0.2),
           nn.ReLU())
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
       self.conn_layer7 = nn.Sequential(nn.Linear(in_features=32,out_features=16),
           nn.Dropout(0.2),
           nn.ReLU())
       self.conn_layer8 = nn.Sequential(nn.Linear(in_features=16,out_features=1))
       self._initialize_weights()
       
   def forward(self,input):
       output = self.conv_layer1(input)
       output = self.a1(output)
       output = self.conv_layer2(output)
       output = self.a2(output)
       output = self.flatten(output)
       output = self.conn_layer1(output)
       output = self.conn_layer2(output)
       output = self.conn_layer3(output)
       output = self.conn_layer4(output)
       output = self.conn_layer5(output)
       output = self.conn_layer6(output)
       output = self.conn_layer7(output)
       output = self.conn_layer8(output)
       output=math.e ** output
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

