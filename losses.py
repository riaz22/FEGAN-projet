import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F


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
    # L1Loss already averages over every element, so no extra /(c*h*w);
    # alpha up-weights the edited region (M==1), which is where generation happens
    loss=self.alpha*self.Criterion(M*generated,M*target)+self.Criterion((1-M)*generated,(1-M)*target)
    return loss

# Perceptual and style losses
class Perceptual_style_Losses(nn.Module):
  def __init__(self):
    super(Perceptual_style_Losses,self).__init__()
    vgg16=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    self.Criterion=nn.L1Loss()
    self.pool1=nn.Sequential(*list(vgg16.features.children())[:5])
    self.pool2=nn.Sequential(*list(vgg16.features.children())[:10])
    self.pool3=nn.Sequential(*list(vgg16.features.children())[:17])

    self.register_buffer('normalization_mean', torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
    self.register_buffer('normalization_std', torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

    self.resize=transforms.Resize((224, 224))

    for param in self.parameters():
      param.requires_grad=False

  def normalize(self,x):
      # inputs live in [-1,1] (tanh range): map to [0,1] then to ImageNet statistics — exactly once
      x=(x+1)/2
      x=(x-self.normalization_mean)/self.normalization_std
      return x

  def gram(self,feat):
    b,c,h,w=feat.size()
    feat=feat.view(b,c,h*w)
    return torch.matmul(feat,feat.transpose(1,2))/(c*h*w)

  def forward(self,target,generated):
      target=self.normalize(self.resize(target))
      generated=self.normalize(self.resize(generated))

      total_percep_loss=0
      total_style_loss=0
      for pool in (self.pool1,self.pool2,self.pool3):
        feat_target=pool(target)
        feat_generated=pool(generated)
        total_percep_loss=total_percep_loss+self.Criterion(feat_generated,feat_target)
        total_style_loss=total_style_loss+self.Criterion(self.gram(feat_generated),self.gram(feat_target))

      return total_percep_loss , total_style_loss


#Total variation Loss
class tv_loss(nn.Module):
    def __init__(self):
        super(tv_loss, self).__init__()
        self.dilation_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        nn.init.constant_(self.dilation_conv.weight, 1.0)
        self.dilation_conv.weight.requires_grad_(False)

    def forward(self, image, mask):
        # mask==1 marks the edited region: dilate the mask itself so the loss
        # covers the hole plus its boundary (not the complement of the hole)
        output_mask = self.dilation_conv(mask)
        dilated_holes = (output_mask != 0).float()
        P = dilated_holes * image
        a = torch.mean(torch.abs(P[:, :, :, 1:] - P[:, :, :, :-1]))
        b = torch.mean(torch.abs(P[:, :, 1:, :] - P[:, :, :-1, :]))

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
  def forward(self,mask,sketch,comp,GT):
      batch_size=GT.size(0)
      # uniform (not gaussian) interpolation factor so samples lie between real and fake
      alpha=torch.rand(batch_size,1,1,1,device=GT.device)
      Igt=torch.cat((GT,mask,sketch),dim=1)
      Icomp=torch.cat((comp,mask,sketch),dim=1).detach()
      interpolated=alpha*Igt+(1-alpha)*Icomp
      interpolated.requires_grad_(True)
      d_interpolated=self.Discriminator(interpolated,five_channels=True)
      #Calculate the gradients
      gradients=torch.autograd.grad(
          outputs=d_interpolated,
          inputs=interpolated,
          grad_outputs=torch.ones_like(d_interpolated),
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
    # hinge loss: the relu stops well-classified samples from dominating the critic update
    comp_loss=torch.mean(F.relu(1+self.discriminator(comp,five_channels=True)))
    gt_loss=torch.mean(F.relu(1-self.discriminator(gt,five_channels=True)))
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
