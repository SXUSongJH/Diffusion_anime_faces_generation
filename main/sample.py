import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(project_root)
os.chdir(project_root)
import torch
import torch.nn as nn
from denoising_diffusion.denoising_model import Unet
from denoising_diffusion.diffusion_process import GaussianDiffusion

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load trained model
denoise_model = Unet(
    dim = 64,
    channels = 3,
    dim_mults=(1, 2, 4, 8, 16)
).to(device)
denoise_model = torch.nn.DataParallel(denoise_model)
denoise_model.load_state_dict(torch.load('../DDPM-Pytorch/results/denoise_model.pth'))
denoise_model.eval()

# define guassian diffusion
Gaussiandiffusion = GaussianDiffusion(
    denoise_model = denoise_model,
    image_size = 128,
    channels = 3,
).to(device)
Gaussiandiffusion = torch.nn.DataParallel(Gaussiandiffusion)

# sample
for n in range(20):
    Gaussiandiffusion.module.get_generated_samples(current_epoch='Done'+str(n), samples_num=25, cols=5)