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
from torchvision.transforms import ToPILImage











def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
dataset=Custom_fegan_dataset('/content/data_sc')
n_epochs = 5
display_step = 50
batch_size = 4
lr_gen=0.001
lr_disc = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model=True
#Create models and initializing weights
generator=DataParallel(Generator(9,hidden_in=64)).to(device)
discriminator=DataParallel(Discriminator()).to(device)
gen_opt = torch.optim.Adam(generator.parameters() , lr=lr_gen, betas=(0.5, 0.999))
disc_opt = torch.optim.Adam(discriminator.parameters() , lr=lr_disc, betas=(0.5, 0.999))

gen_lr_scheduler=torch.optim.lr_scheduler.StepLR(gen_opt,step_size=5,gamma=0.95)
disc_lr_scheduler=torch.optim.lr_scheduler.StepLR(disc_opt,step_size=5,gamma=0.95)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')  # Initialize gated convolution weights with Kaiming initialization
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)  # Initialize batch normalization weights with mean 1 and standard deviation 0.02
        nn.init.constant_(m.bias.data, 0)  # Set batch normalization biases to 0
    elif hasattr(m, 'weight') and 'conv_sigma' in m.weight.__dict__:  # Check if it's a spectral normalized layer
        nn.init.normal_(m.weight.data)  # Initialize spectral normalized layer weights with a normal distribution

pretrained = True
if pretrained:
    pre_dict = torch.load('/kaggle/input/training/sc_FEGAN_2023x32.pth')
    generator.load_state_dict(pre_dict['generator'])
    gen_opt.load_state_dict(pre_dict['gen_opt'])
    discriminator.load_state_dict(pre_dict['discriminator'])
    disc_opt.load_state_dict(pre_dict['disc_opt'])
else:
    generator = generator.apply(weights_init)
    discriminator = discriminator.apply(weights_init)
  def train(save_model=True):
      mean_generator_loss = 0
      mean_discriminator_loss = 0
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
      cur_step = 0
      to_pil = ToPILImage()
      saving='/kaggle/working/'
      for epoch in range(n_epochs):
          # Dataloader returns the batches       
          for generator_input, mask_input, sketch_input, ground_truth in tqdm(dataloader):           
              generator_input=generator_input.to(device)
              mask_input=mask_input.to(device)
              sketch_input=sketch_input.to(device)
              ground_truth=ground_truth.to(device)
              Igt=ground_truth
              
              Igen=generator(generator_input)[0]
              Icomp=gen_to_comp(mask_input,Igen,Igt)
              
              ### Update discriminator  ###
              disc_opt.zero_grad() # Zero out the gradient before backpropagation
              disc_loss = total_disc_loss(discriminator)(mask_input,sketch_input,Icomp,Igt)
              disc_loss.backward(retain_graph=True) # Update gradients
              disc_opt.step() # Update optimizer

              ### Update generator ###
              gen_opt.zero_grad()
              gen_loss= total_gen_loss(discriminator)(mask_input,sketch_input,Igen,Icomp,Igt)
              gen_loss.backward() # Update gradients
              gen_opt.step() # Update optimizer

              # Keep track of the average discriminator loss
              mean_discriminator_loss += disc_loss.item() / display_step
              # Keep track of the average generator loss
              mean_generator_loss += gen_loss.item() / display_step

              ### Visualization Igt , Icomp , Igen ###
              if cur_step % display_step == 0:
                  print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                  show_tensor_images(torch.cat([Igt[0], Icomp[0],Igen[0]]), size=(3, 512, 512))   
                  mean_generator_loss = 0
                  mean_discriminator_loss = 0
                  generated= to_pil(Igen[0])
                  completed=to_pil(Icomp[0])
                  gen_path=os.path.join(saving,'generated_{}.jpg'.format(cur_step))
                  comp_path=os.path.join(saving,'completed_{}.jpg'.format(cur_step))
                  generated.save(gen_path)
                  completed.save(comp_path)
              cur_step += 1
          gen_lr_scheduler.step()
          disc_lr_scheduler.step()
          if save_model:
                      torch.save({
                          'generator': generator.state_dict(),
                          'gen_opt': gen_opt.state_dict(),
                          'discriminator': discriminator.state_dict(),
                          'disc_opt': disc_opt.state_dict()
                      }, '/kaggle/working/'+f"sc_FEGAN{epoch}.pth")
                      generated= to_pil(Igen[0])
                      completed=to_pil(Icomp[0])
                      gen_path=os.path.join(saving,'generated_{}.jpg'.format(epoch))
                      comp_path=os.path.join(saving,'completed_{}.jpg'.format(epoch))
                      generated.save(gen_path)
                      completed.save(comp_path)
  train()
