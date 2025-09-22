import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class Down(nn.Module):
    def __init__(self, c_in, c_out, time_emb_dim=256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(1, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(1, c_out),
            nn.SiLU()
        )
        self.downsample = nn.MaxPool2d(2)
        self.t_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, c_out)
        )
    def forward(self, x, t):
        x = self.convs(x)
        skip = x
        x = self.downsample(x)
        t_emb = self.t_emb(t)[:, :, None, None]
        return x + t_emb, skip


class Up(nn.Module):
    def __init__(self, c_in, c_out, time_emb_dim=256):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.convs = nn.Sequential(
            nn.Conv2d(c_in*2, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(1, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(1, c_out),
            nn.SiLU()
        )
        self.t_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, c_out)
        )
    
    def forward(self, x, x_skip, t):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.convs(x)
        t_emb = self.t_emb(t)[:, :, None, None]
        return x + t_emb
    

class Neutral(nn.Module):
    def __init__(self, c_in, c_out, time_emb_dim=256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(1, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(1, c_out),
            nn.SiLU()
        )
        self.t_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, c_out)
        )
    
    def forward(self, x, t):
        x = self.convs(x)
        t = self.t_emb(t)[:, :, None, None]
        return x + t
    

class SelfAttention(nn.Module):
    def __init__(self, channels, im_size):
        super().__init__()
        self.channels = channels
        self.im_size = im_size
        self.mha = nn.MultiheadAttention(channels, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.LayerNorm(channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels)
        )
    
    def forward(self, x):
        # x is (b, c, H, W). For MHA need a (B, T, C) format
        x = x.view(-1, self.channels, self.im_size * self.im_size).swapaxes(1, 2)  # (B, T, C)
        attn_scores, _ = self.mha(x, x, x)
        attn_scores = attn_scores + x
        out = self.mlp(attn_scores)
        out = out.swapaxes(1, 2).view(-1, self.channels, self.im_size, self.im_size)  # (b, c, H, W)
        return out



class UNet(nn.Module):
    def __init__(self, time_emb_dim=256):
        super().__init__()
        self.inp = nn.Conv2d(3, 64, kernel_size=1)  # (3, 64, 64) -> (64, 64, 64)
        self.down_1 = Down(64, 64)  # (64, 64, 64) -> (64, 32, 32)
        self.down_2 = Down(64, 128)  # (64, 32, 32) -> (128, 16, 16)
        self.sa_1 = SelfAttention(128, 16)
        self.down_3 = Down(128, 256)  # (128, 16, 16) -> (256, 8, 8)

        self.bottleneck_1 = Neutral(256, 512)
        self.sa_2 = SelfAttention(512, 8)
        self.bottleneck_2 = Neutral(512, 512)
        self.sa_3 = SelfAttention(512, 8)
        self.bottleneck_3 = Neutral(512, 256)
        self.sa_4 = SelfAttention(256, 8)

        self.up_1 = Up(256, 128)  # (256, 8, 8) -> (128, 16, 16)
        self.sa_5 = SelfAttention(128, 16)
        self.up_2 = Up(128, 64)  # (128, 16, 16) -> (64, 32, 32)
        self.up_3 = Up(64, 64)  # (64, 32, 32) -> (64, 64, 64)
        self.out = nn.Conv2d(64, 3, kernel_size=1)  # (64, 64, 64) -> (3, 64, 64)
        self.time_emb_dim = time_emb_dim

    def time_embedding(self, t):
        inv_freqs = 1.0 / (1_000 ** (torch.arange(0, self.time_emb_dim, 2, device=device).float() / self.time_emb_dim))
        sin_rep = torch.sin(t.repeat(1, self.time_emb_dim // 2) * inv_freqs)
        cos_rep = torch.cos(t.repeat(1, self.time_emb_dim // 2) * inv_freqs)
        return torch.cat([sin_rep, cos_rep], dim=-1)

    def forward(self, x, t):
        t = t.unsqueeze(-1).float()  # (b,)
        t = self.time_embedding(t)
        y_start = self.inp(x)
        y_1, skip_1 = self.down_1(y_start, t)
        y_2, skip_2 = self.down_2(y_1, t)
        y_2 = self.sa_1(y_2)
        y_3, skip_3 = self.down_3(y_2, t)

        y_4 = self.bottleneck_1(y_3, t)
        y_4 = self.sa_2(y_4)
        y_5 = self.bottleneck_2(y_4, t)
        y_5 = self.sa_3(y_5)
        y_6 = self.bottleneck_3(y_5, t)
        y_6 = self.sa_4(y_6)

        y_7 = self.up_1(y_6, skip_3, t)
        y_7 = self.sa_5(y_7)
        y_8 = self.up_2(y_7, skip_2, t)
        y_9 = self.up_3(y_8, skip_1, t)
        out = self.out(y_9)
        return out
