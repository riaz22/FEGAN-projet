import os
import copy
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.nn import DataParallel
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

from dataset import Custom_fegan_dataset
from models import Generator, Discriminator
from losses import total_gen_loss, total_disc_loss
from utils import gen_to_comp


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def to_pil_image(tensor):
    # tensors live in [-1,1]; ToPILImage expects [0,1]
    return ToPILImage()(((tensor + 1) / 2).clamp(0, 1).cpu())


dataset = Custom_fegan_dataset('/kaggle/input/celebhq-data')
n_epochs = 5
display_step = 50
batch_size = 4
# TTUR: the discriminator learns faster than the generator
lr_gen = 1e-4
lr_disc = 4e-4
ema_decay = 0.999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model = True

#Create models and initializing weights
generator = DataParallel(Generator(9, hidden_in=64)).to(device)
discriminator = DataParallel(Discriminator()).to(device)
gen_opt = torch.optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

gen_lr_scheduler = torch.optim.lr_scheduler.StepLR(gen_opt, step_size=1, gamma=0.95)
disc_lr_scheduler = torch.optim.lr_scheduler.StepLR(disc_opt, step_size=1, gamma=0.95)

# build the losses once: total_gen_loss loads a pretrained VGG16, so constructing
# it inside the training loop would reload VGG from disk on every step
gen_loss_fn = total_gen_loss(discriminator).to(device)
disc_loss_fn = total_disc_loss(discriminator).to(device)


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight_orig'):
        # spectral-norm layers recompute .weight from .weight_orig every forward,
        # so .weight_orig is the tensor that must be initialized
        nn.init.kaiming_normal_(m.weight_orig.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('GroupNorm') != -1 and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


pretrained = False
if pretrained:
    pre_dict = torch.load('/kaggle/input/training/sc_FEGAN_2023x32.pth')
    generator.load_state_dict(pre_dict['generator'])
    gen_opt.load_state_dict(pre_dict['gen_opt'])
    discriminator.load_state_dict(pre_dict['discriminator'])
    disc_opt.load_state_dict(pre_dict['disc_opt'])
else:
    generator = generator.apply(weights_init)
    discriminator = discriminator.apply(weights_init)

# EMA copy of the generator: use it for inference, it gives smoother results
# than the raw generator weights
ema_generator = copy.deepcopy(generator)
for p in ema_generator.parameters():
    p.requires_grad_(False)


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


def train(save_model=True):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0
    saving = '/kaggle/working/'
    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for generator_input, mask_input, sketch_input, ground_truth in tqdm(dataloader):
            generator_input = generator_input.to(device)
            mask_input = mask_input.to(device)
            sketch_input = sketch_input.to(device)
            ground_truth = ground_truth.to(device)
            Igt = ground_truth

            Igen = generator(generator_input)[0]
            Icomp = gen_to_comp(mask_input, Igen, Igt)

            ### Update discriminator ###
            disc_opt.zero_grad()
            disc_loss = disc_loss_fn(mask_input, sketch_input, Icomp, Igt)
            disc_loss.backward()  # comp is detached inside the loss, no retain_graph needed
            disc_opt.step()

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss = gen_loss_fn(mask_input, sketch_input, Igen, Icomp, Igt)
            gen_loss.backward()
            gen_opt.step()
            update_ema(ema_generator, generator, ema_decay)

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization Igt , Icomp , Igen ###
            if cur_step % display_step == 0:
                # raw critic outputs: if |D(real) - D(fake)| keeps growing the
                # discriminator is winning and training is diverging
                with torch.no_grad():
                    d_real = discriminator(torch.cat((Igt, mask_input, sketch_input), dim=1), five_channels=True).mean().item()
                    d_fake = discriminator(torch.cat((Icomp, mask_input, sketch_input), dim=1), five_channels=True).mean().item()
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, "
                      f"Discriminator loss: {mean_discriminator_loss}, D(real): {d_real:.4f}, D(fake): {d_fake:.4f}")
                show_tensor_images(torch.cat([Igt[0], Icomp[0], Igen[0]]), size=(3, 512, 512))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                generated = to_pil_image(Igen[0])
                completed = to_pil_image(Icomp[0])
                gen_path = os.path.join(saving, 'generated_{}.jpg'.format(cur_step))
                comp_path = os.path.join(saving, 'completed_{}.jpg'.format(cur_step))
                generated.save(gen_path)
                completed.save(comp_path)
            cur_step += 1
        gen_lr_scheduler.step()
        disc_lr_scheduler.step()
        if save_model:
            torch.save({
                'generator': generator.state_dict(),
                'ema_generator': ema_generator.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'discriminator': discriminator.state_dict(),
                'disc_opt': disc_opt.state_dict()
            }, '/kaggle/working/' + f"sc_FEGAN{epoch}.pth")
            generated = to_pil_image(Igen[0])
            completed = to_pil_image(Icomp[0])
            gen_path = os.path.join(saving, 'generated_{}.jpg'.format(epoch))
            comp_path = os.path.join(saving, 'completed_{}.jpg'.format(epoch))
            generated.save(gen_path)
            completed.save(comp_path)


if __name__ == '__main__':
    train(save_model=save_model)
