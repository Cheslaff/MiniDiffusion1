import torch
from torch import nn, optim
from torch.nn import functional as F


# TODO: DowSampling, Upsampling, Multihead attention

class DoubleConv(nn.Module):
  def __init__(self, in_c, out_c, mid_c=None, residual=False):
    super().__init__()
    self.residual = residual
    if not mid_c:
      mid_c = out_c
    self.model = nn.Sequential(
        nn.Conv2d(in_c, mid_c, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(1, mid_c),
        nn.GELU(),
        nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1),
        nn.GroupNorm(1, out_c)
    )

  def forward(self, x):
    if self.residual:
      return F.gelu(x + self.model(x))
    else:
      return F.gelu(self.model(x))


class Down(nn.Module):
  def __init__(self):
    # FUCK I DON'T KNOW WHAT TO DO
    # AND I'M TIRED
    # Let it happen (later)

class UNet(nn.Module):
  def __init__(self, ):
    super().__init__()
    # encoder
    self.enc_layer = DoubleConv(3, 64)
    self.down_1 = Down(64, 128)  # (64) -> (32)
    self.sa_1 = MultiheadAttention(128, 32)
    self.down_2 = Down(128, 256)  # (32) -> (16)
    self.sa_2 = MultiheadAttention(256, 32)
    self.down_3 = Down(256, 256)  # (16) -> (8)
    self.sa_3 = MultiheadAttention(256, 32)

    # bottleneck
    self.btln1 = DoubleConv(256, 512)
    self.btln2 = DoubleConv(512, 512)
    self.btln3 = DoubleConv(512, 256)

    self.up_1 = Up(256, 128)  # (8) -> (16)
    self.sa_4 = MultiheadAttention(128, 32)
    self.up_2 = Up(128, 64)  # (16) -> 32
    self.sa_5 = MultiheadAttention(64, 32)
    self.up_3 = Up(64, 64)  # (32) -> (64)
    self.out = nn.Conv2d(64, 3, kernel_size=1)