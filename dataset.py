import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Custom_fegan_dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_input_dir = '/kaggle/input/celebhq-data/image_input/image_input'
        self.color_input_dir = '/kaggle/input/celebhq-data/color_input/color_input'
        self.mask_input_dir = '/kaggle/input/celebhq-data/mask_input/mask_input'
        self.sketch_input_dir = '/kaggle/input/celebhq-data/sketch_input/sketch_input'
        self.noise_input_dir = '/kaggle/input/celebhq-data/noise_input/noise_input'
        self.ground_truth_dir = '/kaggle/input/celebhq-data/ground_truth/ground_truth'

        self.image_input_files = sorted(os.listdir(self.image_input_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.color_input_files = sorted(os.listdir(self.color_input_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.mask_input_files = sorted(os.listdir(self.mask_input_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.sketch_input_files = sorted(os.listdir(self.sketch_input_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.noise_input_files = sorted(os.listdir(self.noise_input_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.ground_truth_files = sorted(os.listdir(self.ground_truth_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def convert_to_grayscale(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray_image

    def __len__(self):
        return len(self.image_input_files)

    def generator_input(self,image_input,mask_input,sketch_input,color_input,noise_input):
          return cv2.merge((image_input,mask_input,sketch_input,color_input,noise_input))


    def __getitem__(self, idx):
        image_input_path = os.path.join(self.image_input_dir, self.image_input_files[idx])
        color_input_path = os.path.join(self.color_input_dir, self.color_input_files[idx])
        mask_input_path = os.path.join(self.mask_input_dir, self.mask_input_files[idx])
        sketch_input_path = os.path.join(self.sketch_input_dir, self.sketch_input_files[idx])
        noise_input_path = os.path.join(self.noise_input_dir, self.noise_input_files[idx])
        ground_truth_path = os.path.join(self.ground_truth_dir, self.ground_truth_files[idx])

        image_input = cv2.imread(image_input_path)
        color_input = cv2.imread(color_input_path)
        mask_input = cv2.imread(mask_input_path)
        sketch_input = cv2.imread(sketch_input_path)
        noise_input = cv2.imread(noise_input_path)
        ground_truth = cv2.imread(ground_truth_path)

        
        image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        color_input = cv2.cvtColor(color_input, cv2.COLOR_BGR2RGB)
        mask_input = cv2.cvtColor(mask_input, cv2.COLOR_BGR2RGB)
        sketch_input = cv2.cvtColor(sketch_input, cv2.COLOR_BGR2RGB)
        noise_input = cv2.cvtColor(noise_input, cv2.COLOR_BGR2RGB)
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)

       # image_input = cv2.rotate(image_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
       # color_input = cv2.rotate(color_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
       # mask_input = cv2.rotate(mask_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #sketch_input = cv2.rotate(sketch_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #noise_input = cv2.rotate(noise_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #ground_truth = cv2.rotate(ground_truth, cv2.ROTATE_90_COUNTERCLOCKWISE)

        generator_input=self.generator_input(image_input,
                                             self.convert_to_grayscale(mask_input),
                                             self.convert_to_grayscale(sketch_input),
                                             color_input,
                                             self.convert_to_grayscale(noise_input))

        mask_input = self.transform(self.convert_to_grayscale(mask_input))
        sketch_input = self.transform(self.convert_to_grayscale(sketch_input))
        ground_truth = self.transform(ground_truth)
        generator_input=self.transform(generator_input)

        return generator_input, mask_input, sketch_input, ground_truth
