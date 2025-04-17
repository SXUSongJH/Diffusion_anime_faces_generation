import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import numpy as np
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from denoising_diffusion.sampling import show_images
from denoising_diffusion.image_process import hf_preprocess
from denoising_diffusion.denoising_model import Unet
from denoising_diffusion.diffusion_process import GaussianDiffusion, train

# load dataset from hugging face
# cifar10_dataset = load_dataset("uoft-cs/cifar10", split='train')
cifar10_dataset = load_from_disk('DDPM-Pytorch\datasets\cifar10')

# show some samples from dataset
image_idx = np.random.randint(0, len(cifar10_dataset), size=24).tolist()
show_images(idx = image_idx, hf_datasets = cifar10_dataset, image_size = 32)

# dataset preprocess
# transform image into tensor
cifar10_dataset_transformed = hf_preprocess(
    cifar10_dataset,
    image_size = 32,
    channels = 3,
    exist_label = True,
    need_label = False
)

# set training device
device = "cuda" if torch.cuda.is_available() else "cpu"

# define denoising model
denoise_model = Unet(
    dim = 32,
    channels = 3,
    dim_mults=(1, 2, 4, 8)
).to(device)

# Use DataParallel to wrap the model for multi-GPU training
denoise_model = torch.nn.DataParallel(denoise_model)

# define diffuion process
Gaussiandiffusion = GaussianDiffusion(
    denoise_model = denoise_model,
    image_size = 32,
    channels = 3,
).to(device)

# Use DataParallel to wrap the diffusion process for multi-GPU training
Gaussiandiffusion = torch.nn.DataParallel(Gaussiandiffusion)

# get noising images and gif
x_start = cifar10_dataset_transformed[1314]['img_tensor'].unsqueeze(0)
Gaussiandiffusion.module.get_noising_images(x_start) # Use .module to access the original model
Gaussiandiffusion.module.get_noising_gif(x_start)

# set dataloader
cifar10_dataloader = DataLoader(
    cifar10_dataset_transformed,
    batch_size=128,
    shuffle=True,
    drop_last=True,
)

# train diffusion model
train(
    Gaussiandiffusion,
    cifar10_dataloader,
    batch_size = 128,
    device = device,
    lr = 1e-4,
    epoch = 3,
    loss_every_epoch = 2,
    sampling_counts = 2
)

# get denoising samples
Gaussiandiffusion.module.get_denoising_samples()

# get denosing gif
Gaussiandiffusion.module.get_denoising_gif()

# get generated samples
Gaussiandiffusion.module.get_generated_samples(current_epoch='Done')

# save model
torch.save(Gaussiandiffusion.module.denoise_model.state_dict(), '../results/denoise_model.pth')
print('Denoise model successfully saved to: ../results/denoise_model.pth')
