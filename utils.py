import torch
import torch.nn.functional as F
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow
from torch import nn


#Icomp generation
def gen_to_comp(mask,gen,gt):
    Icomp=torch.where(mask==1,gen,gt)
    return Icomp





#defining the gate conv and deconv classes
class GateConv(nn.Module):
    def __init__(self, im_chan,cnum, ksize=3, stride=1, rate=1, padding='same', activation='leaky_relu', use_lrn=True):
        super(GateConv, self).__init__()
        self.use_lrn = use_lrn
        self.activation = activation
        self.conv = nn.Conv2d(
            im_chan, cnum, ksize, stride, dilation=rate,
            padding=padding, bias=False)
        self.gate_conv = nn.Conv2d(
            im_chan, cnum, ksize, stride, dilation=rate,
            padding=padding, bias=True)
        self.batch_norm=nn.BatchNorm2d(cnum)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_out = self.conv(x)

        if self.use_lrn:
            x_out =self.batch_norm(x_out)

        if self.activation == 'leaky_relu':
            x_out = F.leaky_relu(x_out)

        g = self.gate_conv(x)
        g = self.sigmoid(g)

        x_out = x_out * g
        return x_out, g
class GateDeconv(nn.Module):
    def __init__(self, input_channels, output_channels,kernel_size=4,stride=2,padding=1):
        super(GateDeconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=True)
        self.deconv_g = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        deconv = self.deconv(x)
        g = self.deconv_g(x)
        g = self.sigmoid(g)

        deconv = deconv * g

        return deconv, g

#spectral normalization convolution
class sn_conv(nn.Module):
  def __init__(self,input_channels,output_channels,kernel_size=5,stride=2,use_act=True):
    super(sn_conv,self).__init__()
    self.use_act=use_act
    self.conv=nn.Conv2d(input_channels,output_channels,kernel_size,stride,padding=1)
    self.conv=nn.utils.spectral_norm(self.conv)
    if self.use_act :
       self.activation=nn.LeakyReLU()
  def forward(self,x):
    x=self.conv(x)
    if self.use_act:
       x=self.activation(x)
    return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = GateConv(in_channels, out_channels, ksize=3, stride=1, padding=1)
        self.conv2 = GateConv(out_channels, out_channels, ksize=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x

        out,_ = self.conv1(x)
        out,_ = self.conv2(out)
        out += residual
        out=self.relu(out)
        return out
class ResidualBlock1(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock1, self).__init__()
        self.conv1 = sn_conv(input_channels, output_channels, kernel_size=3, stride=1)
        self.conv2 = sn_conv(output_channels, output_channels, kernel_size=3, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.conv2(out)
        
        out += residual
        

        return out

