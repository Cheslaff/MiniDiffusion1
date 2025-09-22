import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision

from noise_prediction import UNet
from processes import sample_timestep, add_noise, sample

from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 500
LR = 1e-3
BATCH_SIZE = 128


def plot_images(images):
  plt.figure(figsize=(32, 32))
  plt.imshow(torch.cat([
      torch.cat([i for i in images.cpu()], dim=-1)
  ], dim=-2).permute(1, 2, 0).cpu())
  plt.show()

def save_images(images, path, **kwargs):
  grid = torchvision.utils.make_grid(images, **kwargs)
  ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
  im = Image.fromarray(ndarr)
  im.save(path)


transform = transforms.Compose([
    transforms.Resize(80),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder("MiniDiffusion1\data", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


model = UNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    for images, _ in tqdm(dataloader):
        n = images.size(0)
        images = images.to(device)
        ts = sample_timestep(n).to(device)
        images, noise = add_noise(images, ts)
        noise_pred = model(images, ts)
        loss = loss_fn(noise, noise_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.empty_cache()
    if epoch % 10 == 0:
        print(f"Loss: {loss.item():.4f}")
        images = sample(model, 9)
        save_images(images, f"MiniDiffusion1\samples\sample_epoch{epoch}.png")
        torch.save(model.state_dict(), "MiniDiffusion1\checkpoint.pth")
