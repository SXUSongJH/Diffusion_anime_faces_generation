import os
import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.optim import Adam
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .image_process import reverse_transform
from .sampling import subplots_num


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def alpha_schedule(betas):
    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    return {
        'alphas' : alphas,
        'alphas_cumprod' : alphas_cumprod,
        'alphas_cumprod_prev' : alphas_cumprod_prev,
        'sqrt_recip_alphas' : sqrt_recip_alphas,
        'sqrt_alphas_cumprod' : sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod' : sqrt_one_minus_alphas_cumprod
    }

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)




class GaussianDiffusion(Module):
    def __init__(
        self,
        denoise_model,
        image_size,
        channels,
        timesteps = 1000,
        beta_schedule = 'linear',
        schedule_fn_kwargs = dict(),
        save_dir = 'DDPM-Pytorch/results',
        dpi = 300,
        
    ):
        super().__init__()

        self.denoise_model  = denoise_model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        elif beta_schedule == 'quadratic':
            beta_schedule_fn = quadratic_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        self.betas = betas
        self.alphas_dict = alpha_schedule(betas)
        self.save_dir = save_dir
        self.dpi = dpi

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod = self.alphas_dict['sqrt_alphas_cumprod']
        sqrt_one_minus_alphas_cumprod = self.alphas_dict['sqrt_one_minus_alphas_cumprod']
        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # calculate losses in denoising backward
    def p_losses(self, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas = self.betas
        sqrt_one_minus_alphas_cumprod = self.alphas_dict['sqrt_one_minus_alphas_cumprod']
        sqrt_recip_alphas = self.alphas_dict['sqrt_recip_alphas']
        alphas_cumprod_prev = self.alphas_dict['alphas_cumprod_prev']
        alphas_cumprod = self.alphas_dict['alphas_cumprod']

        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.denoise_model.parameters()).device
        b = shape[0]
        image = torch.randn(shape, device=device)
        images = []
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            image = self.p_sample(image, torch.full((b,), i, device=device, dtype=torch.long), i)
            images.append(image)
        
        return images
    
    def get_noisy_image(self, x_start, t):
        x_noisy = self.q_sample(x_start, t)
        noisy_image = reverse_transform(x_noisy.squeeze())
        return noisy_image
    
    def get_noising_images(
        self,
        x_start,
        t_list = [0, 50, 100, 150, 200, 250],
        save_name = 'noising_images'
    ):
        images = [self.get_noisy_image(x_start, torch.tensor([t])) for t in t_list]
        num_rows, num_cols = 1, len(images)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(8*num_cols, 8*num_rows), dpi=self.dpi)
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            image = images[i]
            ax.axis('off')
            ax.imshow(image, interpolation='nearest')
        plt.tight_layout()
            
        save_dir = os.path.join('..', self.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name + '.png')
        print(f'Saving noising images to: {save_path}')
        plt.savefig(save_path, bbox_inches='tight')
        if os.path.exists(save_path):
            print(f'Noising images successfully saved to: {save_path}')
        else:
            print(f'Failed to save noising images to: {save_path}')
        plt.close(fig)

    def get_noising_gif(self, x_start, save_name = 'noising_process'):
        images = []
        fig, ax = plt.subplots(dpi=self.dpi)
        ax.axis('off')
        for t in range(0, self.timesteps, 2):
            image = self.get_noisy_image(x_start, torch.tensor([t]))
            im = ax.imshow(image, interpolation='nearest')
            plt.tight_layout()
            images.append([im])
        animate = animation.ArtistAnimation(fig, images, interval=50, blit=True, repeat_delay=1000)
        save_dir = os.path.join('..', self.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name + '.gif')
        print(f'Saving noising gif to: {save_path}')
        animate.save(save_path)
        if os.path.exists(save_path):
            print(f'Noising gif successfully saved to: {save_path}')
        else:
            print(f'Failed to save noising gif to: {save_path}')
        plt.close(fig)
    
    def get_denoising_samples(self, time_counts=6, save_name='denosing_samples'):
        image_size = self.image_size
        channels = self.channels
        x_denoising = self.p_sample_loop(shape=(1, channels, image_size, image_size))
        x_index = [int(idx) for idx in torch.linspace(int(self.timesteps / 2), self.timesteps-1, time_counts)]
        x_denoising = [x_denoising[idx] for idx in x_index]
        num_rows, num_cols = 1, time_counts
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(8*num_cols, 8*num_rows), dpi=self.dpi)
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            x =  x_denoising[i].squeeze().cpu()
            image = reverse_transform(x)
            ax.axis('off')
            ax.imshow(image, interpolation='nearest')
        plt.tight_layout()
            
        save_dir = os.path.join('..', self.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name + '.png')
        print(f'Saving denoising images to: {save_path}')
        plt.savefig(save_path, bbox_inches='tight')
        if os.path.exists(save_path):
            print(f'Denoising images successfully saved to: {save_path}')
        else:
            print(f'Failed to save denoising images to: {save_path}')
        plt.close(fig)
    
    def get_denoising_gif(self, save_name='denoising_process'):
        image_size = self.image_size
        channels = self.channels
        x_denoising = self.p_sample_loop(shape=(1, channels, image_size, image_size))
        images = []
        fig, ax = plt.subplots(dpi=self.dpi)
        ax.axis('off')
        for x in x_denoising[::2]:
            image = reverse_transform(x.squeeze().cpu())
            im = ax.imshow(image, interpolation='nearest')
            plt.tight_layout()
            images.append([im])
        animate = animation.ArtistAnimation(fig, images, interval=50, blit=True, repeat_delay=1000)
        save_dir = os.path.join('..', self.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name + '.gif')
        print(f'Saving denoising gif to: {save_path}')
        animate.save(save_path)
        if os.path.exists(save_path):
            print(f'Denoising gif successfully saved to: {save_path}')
        else:
            print(f'Failed to save denoising gif to: {save_path}')
        plt.close(fig)
    
    def get_generated_samples(self, current_epoch, samples_num=48, cols=8, save_name='generated_samples'):
        image_size = self.image_size
        channels = self.channels
        x_denoising = self.p_sample_loop(shape=(samples_num, channels, image_size, image_size))
        x_denoised = x_denoising[-1]
        x_denoised = x_denoised.cpu()
        rows_num, cols_num = subplots_num(samples_num, cols)
        fig, axs = plt.subplots(rows_num, cols_num, figsize=(8*cols_num, 8*rows_num), dpi=self.dpi)
        axs = axs.flatten()
        for i, ax in enumerate(axs[:samples_num]):
            x = x_denoised[i]
            image = reverse_transform(x)
            ax.axis('off')
            ax.imshow(image, interpolation='nearest')
        for ax in axs[samples_num:]:
            ax.axis('off')
        plt.tight_layout()

        save_dir = os.path.join('..', self.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name + f'_epoch{current_epoch}'+'.png')
        print(f'Saving generated images to: {save_path}')
        plt.savefig(save_path, bbox_inches='tight')
        if os.path.exists(save_path):
            print(f'Generated images successfully saved to: {save_path}')
        else:
            print(f'Failed to save generated images to: {save_path}')
        plt.close(fig)
        

def loss_curve(losses, save_dir, save_name='loss'):
    fig, ax = plt.subplots(dpi=100)
    ax.plot(losses, color='blue')
    ax.tick_params(axis='both', direction='in')
    ax.set_xlabel('count')
    ax.set_ylabel('loss')
    ax.legend(['loss curve'])

    save_dir = os.path.join('..', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name + '.png')
    print(f'Saving loss plot to: {save_path}')
    plt.savefig(save_path, bbox_inches='tight')
    
    if os.path.exists(save_path):
        print(f'Loss plot successfully saved to: {save_path}')
    else:
        print(f'Failed to save loss plot to: {save_path}')
    
    plt.close(fig)

def train(
        diffusion,
        dataloader,
        batch_size,
        epoch = 10,
        lr = 1e-3,
        device='cpu',
        loss_every_epoch = 5,
        sampling_counts = 5,
        print_loss = True,
        draw_loses_plot = True,
        save_dir = 'DDPM-Pytorch/results',
    ):

    loss_batch_index = [int(idx) for idx in torch.linspace(0, len(dataloader)-1, loss_every_epoch)]
    sample_epoch_index = [int(idx) for idx in torch.linspace(1, epoch, sampling_counts)]
    print('loss batch index:', loss_batch_index)
    print('samples epoch index:', sample_epoch_index)
    if draw_loses_plot:
        losses = []
    
    optimizer = Adam(diffusion.module.denoise_model.parameters(), lr=lr)

    print('-------------------start training!---------------------')
    print('Using devices:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    print('Model is on device:', next(diffusion.parameters()).device)
    for epoch in range(1,epoch+1):
        print(f'-------------------Epoch {epoch}----------------------')
        for step, batch in tqdm(enumerate(dataloader), total = len(dataloader), desc='batches'):
            optimizer.zero_grad()
            batch = batch['img_tensor'].to(device)
            t = torch.randint(0, diffusion.module.timesteps, (batch_size,), device=device).long()
            loss = diffusion.module.p_losses(batch, t, loss_type='huber')
            loss.backward()
            optimizer.step()

            if (step in loss_batch_index) and print_loss:
                print("Loss:", loss.item())
            if draw_loses_plot:
                losses.append(loss.item())
        print('-----------------------------------------------------------')
        
        if epoch in sample_epoch_index:
            diffusion.module.get_generated_samples(epoch)
    
    if draw_loses_plot:
        loss_curve(losses, save_dir=save_dir)