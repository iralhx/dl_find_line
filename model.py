import torch
import torch.nn as nn
import math
from transformersmodel import *
import resnet


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
        self.mind_k =(max_k+min_k)/2
        self.weight = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()
 
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, input):
        output = self.mind_k + input * self.weight
        return output

class ConModel(nn.Module):
    def __init__(self):
        super(ConModel,self).__init__()
        layer1_channels=16
        self.conv_layer1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=layer1_channels,kernel_size=3,stride=2,padding=3//2),
            nn.BatchNorm2d(layer1_channels),
            nn.ReLU()
        )#layer1_channels*128*128
        layer2_channels=32
        self.conv_layer2=nn.Sequential(
            nn.Conv2d(in_channels=layer1_channels,out_channels=layer2_channels,kernel_size=3,stride=2,padding=3//2),
            nn.BatchNorm2d(layer2_channels),
            nn.ReLU()
        )#layer2_channels*64*64
        layer3_channels=64
        self.conv_layer3=nn.Sequential(
            nn.Conv2d(in_channels=layer2_channels,out_channels=layer3_channels,kernel_size=3,stride=2,padding=3//2),
            nn.BatchNorm2d(layer3_channels),
            nn.ReLU()
        )#layer3_channels*64*64
        layer4_channels=128
        self.conv_layer3=nn.Sequential(
            nn.Conv2d(in_channels=layer4_channels,out_channels=layer4_channels,kernel_size=3,stride=2,padding=3//2),
            nn.BatchNorm2d(layer4_channels),
            nn.ReLU()
        )#layer3_channels*32*32
        self.res1=resnet.ResnetBasicBlock(layer3_channels,layer3_channels) 
        layer4_channels=32
        self.res2=resnet.ResnetBasicBlock(layer3_channels,layer4_channels,2) 
        self.res3=resnet.ResnetBasicBlock(layer4_channels,layer4_channels) 
        # 32*32
        self.res4=resnet.ResnetBasicBlock(layer4_channels,layer4_channels,2) 
        #16*16
        self.flatten = Flatten()
        self.conn_layer1 = nn.Sequential(nn.Linear(in_features=layer4_channels*16*16,out_features=1024),
            nn.Dropout(0.5),
            nn.ReLU())
        self.conn_layer2 = nn.Sequential(nn.Linear(in_features=1024,out_features=256),
            nn.Dropout(0.5),
            nn.ReLU())
        self.tr1=TransformerLayer(256,256)
        self.conn_layer3 = nn.Sequential(nn.Linear(in_features=256,out_features=1))
        #    self.kk = kk(0.6,1)
        self._initialize_weights()
       
    def forward(self,input):
        output = self.conv_layer1(input)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = self.res1(output)
        output = self.res2(output)
        output = self.res3(output)
        output = self.res4(output)
        output = self.flatten(output)
        output = self.conn_layer1(output)
        output = self.conn_layer2(output)
        output = self.tr1(output)
        output = self.conn_layer3(output)
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

class ResModel(nn.Module):
    def __init__(self):
        super(ResModel,self).__init__()
        self.res1=resnet.ResnetBasicBlock(1,64) 
        # 256*256
        self.res2=resnet.ResnetBasicBlock(64,128)
        # 256*256
        self.res3=resnet.ResnetBasicBlock(128,256)
        # 256*256
        self.res4=resnet.ResnetBasicBlock(256,512)
        # 256*256
        self.res5=resnet.ResnetBasicBlock(512,256,2)
        # 128*128
        self.res6=resnet.ResnetBasicBlock(256,128,2)
        # 64*64
        self.res7=resnet.ResnetBasicBlock(128,32,2)
        # 32*32
        self.res8=resnet.ResnetBasicBlock(32,16,2)
        # 16*16
       
        self.flatten = Flatten()
        head=1024
        self.tr1=TransformerLayer(16*16*16,head)
        # self.conn_layer1 = nn.Sequential(nn.Linear(in_features=16*16*16,out_features=1),
        #     nn.Dropout(0.2),
        #     nn.ReLU())
        self.conn_layer1 = nn.Sequential(nn.Linear(in_features=16*16*16,out_features=1))
       
    def forward(self,input):
        output = self.res1(input)
        output = self.res2(output)
        output = self.res3(output)
        output = self.res4(output)
        output = self.res5(output)
        output = self.res6(output)
        output = self.res7(output)
        output = self.res8(output)
        output = self.flatten(output)
        output = self.tr1(output)
        output = self.conn_layer1(output)
        # output = self.kk(output)
        return output

class ResModel1(nn.Module):
    def __init__(self):
        super(ResModel1,self).__init__()
        self.res1=resnet.ResnetBasicBlock(1,64) 
        # 256*256
        self.res2=resnet.ResnetBasicBlock(64,128)
        # 256*256
        self.res3=resnet.ResnetBasicBlock(128,256)
        # 256*256
        self.res4=resnet.ResnetBasicBlock(256,512)
        # 256*256
        self.res5=resnet.ResnetBasicBlock(512,256,2)
        # 128*128
        self.res6=resnet.ResnetBasicBlock(256,128,2)
        # 64*64
        self.res7=resnet.ResnetBasicBlock(128,32,2)
        # 32*32
        self.res8=resnet.ResnetBasicBlock(32,16,2)
        # 16*16
       
        self.flatten = Flatten()
        head=1024
        self.tr1=TransformerLayer(16*16*16,head)
        self.conn_layer1 = nn.Sequential(nn.Linear(in_features=16*16*16,out_features=1024),
            nn.Dropout(0.2),
            nn.ReLU())
        self.conn_layer2 = nn.Sequential(nn.Linear(in_features=1024,out_features=1))
        # self.kk=kk(0.5,1)
    def forward(self,input):
        output = self.res1(input)
        output = self.res2(output)
        output = self.res3(output)
        output = self.res4(output)
        output = self.res5(output)
        output = self.res6(output)
        output = self.res7(output)
        output = self.res8(output)
        output = self.flatten(output)
        output = self.tr1(output)
        output = self.conn_layer1(output)
        output = self.conn_layer2(output)
        # output = self.kk(output)
        return output

class ResModelSmall(nn.Module):
    def __init__(self):
        super(ResModelSmall,self).__init__()
        self.res1=resnet.ResnetBasicBlock(1,64) 
        # 256*256
        self.res2=resnet.ResnetBasicBlock(64,128)
        # 256*256
        self.res3=resnet.ResnetBasicBlock(128,256)
        # 256*256
        self.res4=resnet.ResnetBasicBlock(256,512,2)
        # 128*128
        self.res5=resnet.ResnetBasicBlock(512,256,2)
        # 128*128
        self.res6=resnet.ResnetBasicBlock(256,128,2)
        # 64*64
        self.res7=resnet.ResnetBasicBlock(128,32,2)
        # 32*32
        self.res8=resnet.ResnetBasicBlock(32,16,2)
        # 8*8
        self.flatten = Flatten()
        self.conn_layer1 = nn.Sequential(nn.Linear(in_features=16*64,out_features=512),
            nn.Dropout(0.2),
            nn.ReLU())
        self.conn_layer2 = nn.Sequential(nn.Linear(in_features=512,out_features=1))
       
    def forward(self,input):
        output = self.res1(input)
        output = self.res2(output)
        output = self.res3(output)
        output = self.res4(output)
        output = self.res5(output)
        output = self.res6(output)
        output = self.res7(output)
        output = self.res8(output)
        output = self.flatten(output)
        output = self.conn_layer1(output)
        output = self.conn_layer2(output)
        # output = self.kk(output)
        return output
    


class FullConModel(nn.Module):
    def __init__(self):
        super(FullConModel,self).__init__()

        self.conv_layer1=self._make_down_layer(1,8)#128
        self.conv_layer2=self._make_down_layer(8,16)#64
        self.conv_layer3=self._make_down_layer(16,32)#32
        self.conv_layer4=self._make_down_layer(32,64)#16
        self.conv_layer5=self._make_down_layer(64,128)#8
        self.conv_layer6=self._make_up_layer(128,64)#16
        self.conv_layer7=self._make_up_layer(64,32)#32
        self.conv_layer8=self._make_up_layer(32,16)#64
        self.conv_layer9=self._make_up_layer(16,8)#128
        self.conv_layer10=self._make_up_layer(8,1)#256
    

    def _make_down_layer(self,intput,output,kernel_size=3,stride=2):
        layer=nn.Sequential(
            nn.Conv2d(in_channels=intput,out_channels=output,kernel_size=kernel_size
                        ,stride=stride,padding=kernel_size//stride),
            nn.BatchNorm2d(output),
            nn.ReLU()
        )
        return layer
    
    def _make_up_layer(self,intput,output,kernel_size=3,stride=2):
        layer=nn.Sequential(
            nn.ConvTranspose2d(in_channels=intput,out_channels=output,kernel_size=kernel_size
                        ,stride=stride,padding=kernel_size//stride,output_padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU()
        )
        return layer



    def forward(self,input):
        o1 = self.conv_layer1(input)
        o2 = self.conv_layer2(o1)
        o3 = self.conv_layer3(o2)
        o4 = self.conv_layer4(o3)
        o5 = self.conv_layer5(o4)
        o6 = self.conv_layer6(o5)
        o6=o4+o6
        o7 = self.conv_layer7(o6)
        o7=o3+o7
        o8 = self.conv_layer8(o7)
        o8=o2+o8
        o9 = self.conv_layer9(o8)
        o9=o1+o9
        output = self.conv_layer10(o9)
        return output
   
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
