import torch
import torch.nn as nn
import math
from transformersmodel import *
import resnet
from unit import *


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
        self.c1=C3(1,32)
        self.c2=C3(32,64)
        self.c3=C3(64,32)
        self.conv_layer1=self._make_down_layer(32,64)#128
        self.conv_layer2=self._make_down_layer(64,128)#64
        self.conv_layer3=self._make_down_layer(128,128)#32
        self.conv_layer4=self._make_down_layer(128,128)#16
        self.conv_layer5=self._make_down_layer(128,128)#8
        self.conv_layer6=self._make_up_layer(128,64)#16
        self.conv_layer7=self._make_up_layer(64,32)#32
        self.conv_layer8=self._make_up_layer(32,16)#64
        self.conv_layer9=self._make_up_layer(16,8)#128
        self.conv_layer10=self._make_up_layer(8,1,activa='Sigmoid')#256
    

    def _make_down_layer(self,intput,output,kernel_size=7,stride=2):
        layer=nn.Sequential(
            nn.Conv2d(in_channels=intput,out_channels=output,kernel_size=kernel_size
                        ,stride=stride,padding=kernel_size//stride),
            nn.BatchNorm2d(output),
            nn.ReLU()
        )
        return layer
    
    def _make_up_layer(self,intput,output,kernel_size=3,stride=2,activa='Relu'):
        if activa=='Relu':
            layer=nn.Sequential(
                nn.ConvTranspose2d(in_channels=intput,out_channels=output,kernel_size=kernel_size
                            ,stride=stride,padding=kernel_size//stride,output_padding=1),
                nn.BatchNorm2d(output),
                nn.ReLU()
            )
        elif activa=='Sigmoid':
            layer=nn.Sequential(
                nn.ConvTranspose2d(in_channels=intput,out_channels=output,kernel_size=kernel_size
                            ,stride=stride,padding=kernel_size//stride,output_padding=1),
                nn.BatchNorm2d(output),
                nn.Sigmoid()
            )
        return layer



    def forward(self,input):
        o1 = self.c1(input)
        o2 = self.c2(o1)
        o3 = self.c3(o2)
        # o3=o1+o3



        o4 = self.conv_layer1(o3)
        o5 = self.conv_layer2(o4)
        o6 = self.conv_layer3(o5)
        o7 = self.conv_layer4(o6)
        o8 = self.conv_layer5(o7)
        o9 = self.conv_layer6(o8)
        # o9=o7+o9
        o10 = self.conv_layer7(o9)
        # o10=o6+o10
        o11 = self.conv_layer8(o10)
        # o11=o5+o11
        o12 = self.conv_layer9(o11)
        # o12=o4+o12
        output = self.conv_layer10(o12)
        return output



import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 定义编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 定义解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 最后的1x1卷积层用于产生最终的输出
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        x1 = self.encoder(x)

        # 解码器部分
        x2 = self.decoder(x1)

        # 最后的1x1卷积层，输出最终的结果
        output = self.final_conv(x2)

        return output

