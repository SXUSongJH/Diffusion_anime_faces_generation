import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(project_root)
os.chdir(project_root)
import numpy as np
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
from denoising_diffusion.sampling import show_images
from denoising_diffusion.image_process import hf_preprocess
from denoising_diffusion.denoising_model import Unet
from denoising_diffusion.diffusion_process import GaussianDiffusion, train

# load dataset from hugging face
# animefaces_dataset = load_dataset("jlbaker361/anime_faces_dim_128_50k", split='train')
animefaces_dataset = load_from_disk('datasets/anime_faces')

# show some samples from dataset
image_idx = np.random.randint(0, len(animefaces_dataset), size=20).tolist()
show_images(idx = image_idx, hf_datasets = animefaces_dataset, image_size = 128, cols_num=5)

# dataset preprocess
# transform image into tensor
animefaces_dataset_transformed = hf_preprocess(
    animefaces_dataset,
    image_size = 128,
    channels = 3,
    exist_label = False,
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
    image_size = 128,
    channels = 3,
).to(device)

# Use DataParallel to wrap the diffusion process for multi-GPU training
Gaussiandiffusion = torch.nn.DataParallel(Gaussiandiffusion)

# get noising images and gif
x_start = animefaces_dataset_transformed[528]['img_tensor'].unsqueeze(0)
Gaussiandiffusion.module.get_noising_images(x_start, t_list=[0,100,150,200,300]) # Use .module to access the original model
Gaussiandiffusion.module.get_noising_gif(x_start)

# set dataloader
animefaces_dataloader = DataLoader(
    animefaces_dataset_transformed,
    batch_size=4,
    shuffle=True,
    drop_last=True,
)

# train diffusion model
train(
    Gaussiandiffusion,
    animefaces_dataloader,
    batch_size = 4,
    device = device,
    lr = 1e-4,
    epoch = 5,
    loss_every_epoch = 3,
    sampling_counts = 2
)

# get denoising samples
Gaussiandiffusion.module.get_denoising_samples()

# get denosing gif
Gaussiandiffusion.module.get_denoising_gif()

# get generated samples
Gaussiandiffusion.module.get_generated_samples(current_epoch='Done')

# save model
torch.save(Gaussiandiffusion.module.denoise_model.state_dict(), '../DDPM-Pytorch/results/denoise_model.pth')
print('Denoise model successfully saved to: ../results/denoise_model.pth')

# Ensure to clean up CUDA memory after training
torch.cuda.empty_cache()