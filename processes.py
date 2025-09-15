import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


timesteps = 1_000
device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = 64

def sample_timestep(n):
  return torch.randint(1, timesteps, size=(n,))

beta_start = 0.0001
beta_end = 0.02

betas = torch.linspace(start=beta_start, end=beta_end, steps=timesteps)  # (t,)
alphas = 1 - betas  # (t,)
alphas_hat = torch.cumprod(alphas, dim=0)  # (t,)

def add_noise(x, t):
  # x is (B, C, H, W), t is (B,)
  sqrt_alpha_hat = torch.sqrt(alphas_hat[t]).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
  sqrt_one_minus_alpha_hat = torch.sqrt(1 - alphas_hat[t]).view(-1, 1, 1, 1)
  eps = torch.randn_like(x)

  # add noise to image and return both noisy image and noise
  return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

def sample(model, n):
  model.eval()
  with torch.no_grad():
    x = torch.randn(n, 3, img_size, img_size)
    for i in tqdm(reversed(range(1, timesteps))):
      t = (torch.ones(n) * i).long().to(device)
      alpha_t = alphas[t].view(-1, 1, 1, 1)
      alpha_hat_t = alphas_hat[t].view(-1, 1, 1, 1)
      beta_t = betas[t].view(-1, 1, 1, 1)

      noise_pred = model(x, t)
      if i > 1:
        noise = torch.randn_like(x)
      else:
        noise = torch.zeros_like(x)

      x = 1 / torch.sqrt(alpha_t) * (x.detach() - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * noise_pred) + torch.sqrt(beta_t) * noise
  model.train()
  x = (x.clamp(-1, 1) + 1) / 2
  x = (x * 255).type(torch.uint8)
  return x
