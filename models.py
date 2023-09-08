import torch
import cv2
from torch import nn
#from torchsummary import summary
from tqdm.auto import tqdm
from torch.nn import DataParallel
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import GateConv,sn_conv,GateDeconv,ResidualBlock,ResidualBlock1



#generator
class Generator(nn.Module):
   def __init__(self,input_channels,hidden_in=32,kernel_size=3,stride=2):
      super(Generator,self).__init__()
      #encoder
      self.block1=GateConv(input_channels,hidden_in,kernel_size,stride,padding=1,use_lrn=False)
      self.block2=GateConv(hidden_in,2*hidden_in,kernel_size,stride,padding=1)
      self.block3=GateConv(2*hidden_in,4*hidden_in,kernel_size,stride,padding=1)
      self.block4=GateConv(4*hidden_in,8*hidden_in,kernel_size,stride,padding=1)
      self.block5=GateConv(8*hidden_in,8*hidden_in,kernel_size,stride,padding=1)
      self.block6=GateConv(8*hidden_in,8*hidden_in,kernel_size,stride,padding=1)
      self.block7=GateConv(8*hidden_in,8*hidden_in,kernel_size,stride,padding=1)
      #dilated_conv
      self.d_block1=ResidualBlock(8*hidden_in,8*hidden_in,kernel_size)
      self.d_block2=ResidualBlock(8*hidden_in,8*hidden_in,kernel_size)
      self.d_block3=ResidualBlock(8*hidden_in,8*hidden_in,kernel_size)
      self.d_block4=ResidualBlock(8*hidden_in,8*hidden_in,kernel_size)
      self.d_block5=ResidualBlock(8*hidden_in,8*hidden_in,kernel_size)
      self.d_block6=ResidualBlock(8*hidden_in,8*hidden_in,kernel_size)


      #decoder
      self.de_block1=GateDeconv(8*hidden_in,8*hidden_in)
      self.de_block2=GateDeconv(8*hidden_in,8*hidden_in)
      self.de_block3=GateDeconv(8*hidden_in,8*hidden_in)
      self.de_block4=GateDeconv(8*hidden_in,4*hidden_in)
      self.de_block5=GateDeconv(4*hidden_in,2*hidden_in)
      self.de_block6=GateDeconv(2*hidden_in,hidden_in)
      self.de_block7=GateDeconv(hidden_in,3)
      #gate conv for the decoders
      self.gate8=GateConv(16*hidden_in,8*hidden_in)
      self.gate9=GateConv(16*hidden_in,8*hidden_in)
      self.gate10=GateConv(16*hidden_in,8*hidden_in)
      self.gate11=GateConv(8*hidden_in,4*hidden_in)
      self.gate12=GateConv(4*hidden_in,2*hidden_in)
      self.gate13=GateConv(2*hidden_in,hidden_in)
      self.gate14=GateConv(12,3,activation=None,use_lrn=False)
   def forward(self,x):
        x_in=x.clone()
        x1,mask1=self.block1(x)
        x2,mask2=self.block2(x1)
        x3,mask3=self.block3(x2)
        x4,mask4=self.block4(x3)
        x5,mask5=self.block5(x4)
        x6,mask6=self.block6(x5)
        x7,mask7=self.block7(x6)


        x7=self.d_block1(x7)
        x7=self.d_block2(x7)
        x7=self.d_block3(x7)
        x7=self.d_block4(x7)
        x7=self.d_block5(x7)
        x7=self.d_block6(x7)



        x8 ,_=self.de_block1(x7)
        x8=torch.cat([x6, x8], axis=1)
        x8,mask8=self.gate8(x8)


        x9,_=self.de_block2(x8)
        x9=torch.cat([x5,x9],axis=1)
        x9,mask9=self.gate9(x9)

        x10,_=self.de_block3(x9)
        x10=torch.cat([x4,x10],axis=1)
        x10,mask10=self.gate10(x10)

        x11,_=self.de_block4(x10)
        x11=torch.cat([x3,x11],axis=1)
        x11,mask11=self.gate11(x11)

        x12,_=self.de_block5(x11)
        x12=torch.cat([x2,x12],axis=1)
        x12,mask12=self.gate12(x12)

        x13,_=self.de_block6(x12)
        x13=torch.cat([x1,x13],axis=1)
        x13,mask13=self.gate13(x13)

        x14,_=self.de_block7(x13)
        x14=torch.cat([x_in,x14],axis=1)
        x14,mask14=self.gate14(x14)

        x14=torch.tanh(x14)


        return x14,mask14


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64, kernel_size=3, stride=2, padding='same'):
        super(Discriminator, self).__init__()
        self.conv1 = sn_conv(input_channels, hidden_channels)
        self.conv1_x = sn_conv(input_channels + 2, hidden_channels)
        self.conv2 = sn_conv(hidden_channels, 2 * hidden_channels)
        self.residual_blocks = nn.Sequential(
            ResidualBlock1(2 * hidden_channels, 2 * hidden_channels),
            ResidualBlock1(2 * hidden_channels, 2 * hidden_channels),
            ResidualBlock1(2 * hidden_channels, 2 * hidden_channels),
            ResidualBlock1(2 * hidden_channels, 2 * hidden_channels)
        )
        self.conv3 = sn_conv(2 * hidden_channels, 4 * hidden_channels)
        self.conv4 = sn_conv(4 * hidden_channels, 4 * hidden_channels)
        self.conv5 = sn_conv(4 * hidden_channels, 4 * hidden_channels)
        self.conv6 = sn_conv(4 * hidden_channels, 4 * hidden_channels, use_act=False)

    def forward(self, x, five_channels=False):
        if five_channels:
            x = self.conv1_x(x)
        else:
            x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual_blocks(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), -1)
        return x
