import torch
import cv2
from torch import nn
#from torchsummary import summary
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


#WGAN Generator loss
class Gen_adv_loss(nn.Module):
  def __init__(self,discriminator,epsilon=0.001,beta=0.001):
    super(Gen_adv_loss,self).__init__()
    self.epsilon=epsilon
    self.beta=beta
    self.discriminator=discriminator
  
  def forward(self,mask,sketch,comp,gt):
    comp=torch.cat((comp,mask,sketch),dim=1)
    gt=torch.cat((gt,mask,sketch),dim=1)
    loss_comp=-torch.mean(self.discriminator(comp,five_channels=True))
    loss_gt=torch.mean(self.discriminator(gt,five_channels=True)**2)

    return self.beta*loss_comp+self.epsilon*loss_gt

# Per pixel Loss
class Per_pixel_loss(nn.Module):
  def __init__(self,alpha):
    super(Per_pixel_loss,self).__init__()
    self.Criterion=nn.L1Loss()
    self.alpha=alpha

  def forward(self,M,generated,target):
    generated=generated.float()
    target=target.float()
    c,h,w=target.size(1),target.size(2),target.size(3)
    loss=(self.Criterion(M*generated,M*target))/(c*h*w)+self.alpha*(self.Criterion((1-M)*generated,(1-M)*target))/(c*h*w)
    return loss

# Perceptual and style losses
class Perceptual_style_Losses(nn.Module):
  def __init__(self):
    super(Perceptual_style_Losses,self).__init__()
    self.vgg16=models.vgg16(pretrained=True)

    
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.Criterion=nn.L1Loss()
    self.pool1=nn.Sequential(*list(self.vgg16.features.children())[:5])#.to(self.device)
    self.pool2=nn.Sequential(*list(self.vgg16.features.children())[:10])#.to(self.device)
    self.pool3=nn.Sequential(*list(self.vgg16.features.children())[:17])#.to(self.device)
    self.pool1=self.pool1.to(self.device)
    self.pool2=self.pool2.to(self.device)
    self.pool3=self.pool3.to(self.device)

    self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
    self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

    self.transform=transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.ToTensor(),
        transforms.Normalize(self.normalization_mean, self.normalization_std)
    ])

    for param in self.vgg16.parameters():
      param.requires_grad=False
  def normalize(self,x):
      x=(x-0.5)/0.5
      x = (x - self.normalization_mean.view(1, -1, 1, 1)) / self.normalization_std.view(1, -1, 1, 1)
      return x

  def style_process(self,x,y):
    x=x.view(x.size(0),x.size(1),-1)
    x_transpose=x.transpose(1,2)
    c,h,w=y.size(1),y.size(2),y.size(3)
    y=y.view(y.size(0),y.size(1),-1)
    y_transpose=y.transpose(1,2)
    
    
    return self.Criterion(torch.matmul(x,x_transpose),torch.matmul(y,y_transpose))/(c*h*w*c*c)

  def forward(self,target,generated):
      target=self.transform(target)
      generated=self.transform(generated)
      target=self.normalize(target)
      generated=self.normalize(generated)
      #percetual  loss
      loss_percep1=self.Criterion(self.pool1(generated),self.pool1(target))/(self.pool1(target).size(1)*self.pool1(target).size(2)*self.pool1(target).size(3))
      loss_percep2=self.Criterion(self.pool2(generated),self.pool2(target))/(self.pool2(target).size(1)*self.pool2(target).size(2)*self.pool2(target).size(3))
      loss_percep3=self.Criterion(self.pool3(generated),self.pool3(target))/(self.pool3(target).size(1)*self.pool3(target).size(2)*self.pool3(target).size(3))
      total_percep_loss=loss_percep1+loss_percep2+loss_percep3

      #stype loss


      loss_style1=self.style_process(self.pool1(generated),self.pool1(target))
      loss_style2=self.style_process(self.pool2(generated),self.pool2(target))
      loss_style3=self.style_process(self.pool3(generated),self.pool3(target))
      total_style_loss=loss_style1+loss_style2+loss_style3




      return total_percep_loss , total_style_loss


#Total variation Loss
class tv_loss(nn.Module):
    def __init__(self):
        super(tv_loss, self).__init__()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dilation_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.dilation_conv=self.dilation_conv.to(self.device)
        nn.init.constant_(self.dilation_conv.weight, 1.0)

    def forward(self, image, mask):
        output_mask = self.dilation_conv(1 - mask)
        output_mask=output_mask.to(self.device)
        dilated_holes = (output_mask != 0).float()
        P = dilated_holes * image
        c,h,w=image.size(1),image.size(2),image.size(3)
        a = torch.mean(torch.abs(P[:, :, :, 1:] - P[:, :, :, :-1]))/(c*h*w)
        b = torch.mean(torch.abs(P[:, :, 1:, :] - P[:, :, :-1, :]))/(c*h*w)

        total_variation_loss = a + b
        return total_variation_loss

# Generator total loss
class total_gen_loss(nn.Module):
  def __init__(self,discriminator,sigma=0.05,gamma=120,v=0.1):
    super(total_gen_loss,self).__init__()
    self.discriminator=discriminator
    self.sigma=sigma
    self.gamma=gamma
    self.v=v
    


    self.gen_adv_loss=Gen_adv_loss(self.discriminator)
    self.per_pixel_loss=Per_pixel_loss(2)
    self.percept_style_losses=Perceptual_style_Losses()
    self.tv_loss=tv_loss()

  def forward(self,mask,sketch,gen,comp,gt):
    #Gen_adv_loss
    adversarial_loss=self.gen_adv_loss(mask,sketch,comp,gt)
    #Per pixel loss
    per_pixel_loss=self.per_pixel_loss(mask,gen,gt)
    #Perceptual and styles losses 
    percep_gen_gt , style_gen_gt= self.percept_style_losses(gt,gen)
    percep_comp_gt , style_comp_gt= self.percept_style_losses(gt,comp)

    total_percept_loss= percep_gen_gt + percep_comp_gt
    total_style_loss= style_gen_gt + style_comp_gt
    
    # total variation loss
    tv_loss=self.tv_loss(comp,mask)


    #total loss

    total_generator_loss=per_pixel_loss + self.sigma * total_percept_loss + adversarial_loss + self.gamma * total_style_loss + self.v * tv_loss

    return total_generator_loss


'--------------------------------------------------------------------------------------------------------------------------------------------------------------'

#Gradient Penalty loss
class Gradient_penalty_loss(nn.Module):
  def __init__(self,Discriminator):
    super(Gradient_penalty_loss,self).__init__()
    self.Discriminator=Discriminator
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  def forward(self,mask,sketch,comp,GT):
      batch_size=GT.size(0)
      alpha=torch.randn(batch_size,1,1,1)
      alpha=alpha.to(self.device)
      Igt=torch.cat((GT,mask,sketch),dim=1)
      Icomp=torch.cat((comp,mask,sketch),dim=1).detach()
      interpolated=alpha*Igt+(1-alpha)*Icomp
      interpolated.requires_grad_(True)
      #Calculate the gradients
      gradients=torch.autograd.grad(
          outputs=self.Discriminator(interpolated,five_channels=True),
          inputs=interpolated,
          grad_outputs=torch.ones_like(self.Discriminator(interpolated,five_channels=True)),
          create_graph=True,
          retain_graph=True,
          only_inputs=True,
      )[0]

      gradients=gradients*mask
      gradients=gradients.view(gradients.size(0),-1)
      gradients_penalty=((gradients.norm(2,dim=1)-1)**2).mean()

      return gradients_penalty
class disc_adv_loss(nn.Module):
  def __init__(self,discriminator):
    super(disc_adv_loss,self).__init__()
    self.discriminator=discriminator
  def forward(self,mask,sketch,comp,gt):
    comp=torch.cat((comp,mask,sketch),dim=1).detach()
    gt=torch.cat((gt,mask,sketch),dim=1)
    comp_loss=torch.mean(1+self.discriminator(comp,five_channels=True))
    gt_loss=torch.mean(1-self.discriminator(gt,five_channels=True))
    return comp_loss+gt_loss
#WGAN discriminator loss (Hinge Loss function)

class total_disc_loss(nn.Module):
  def __init__(self,discriminator,theta=10):
    super(total_disc_loss,self).__init__()
    self.theta=theta
    self.discriminator=discriminator
    self.GP_loss=Gradient_penalty_loss(self.discriminator)
    self.adv_loss=disc_adv_loss(self.discriminator)


  def forward(self,mask,sketch,comp,gt):
    
    # Gradient penalty loss
    gp_loss=self.GP_loss(mask,sketch,comp,gt)

    # Adversarial loss
    adv_loss=self.adv_loss(mask,sketch,comp,gt)

    #total loss
    total_disc_loss=adv_loss + self.theta*gp_loss

    return total_disc_loss
