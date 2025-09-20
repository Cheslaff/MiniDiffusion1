import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


class Down(nn.Module):
    def __init__(self, c_in, c_out):
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
    
    def forward(self, x, t):
        # t is ignored for now
        x = self.convs(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, c_in, c_out):
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
    
    def forward(self, x, x_skip, t):
        # t is ignored for now
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.convs(x)
        return x
    

class Neutral(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(1, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(1, c_out),
            nn.SiLU()
        )
    
    def forward(self, x, t):
        x = self.convs(x)
        return x
    

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_1 = Down(3, 64)  # (3, 64, 64) -> (64, 32, 32)
        self.down_2 = Down(64, 128)  # (64, 32, 32) -> (128, 16, 16)
        self.down_3 = Down(128, 256)  # (128, 16, 16) -> (256, 8, 8)

        self.bottleneck_1 = Neutral(256, 512)
        self.bottleneck_2 = Neutral(512, 512)
        self.bottleneck_3 = Neutral(512, 256)

        self.up_1 = Up(256, 128)  # (256, 8, 8) -> (128, 16, 16)
        self.up_2 = Up(128, 64)  # (128, 16, 16) -> (64, 32, 32)
        self.up_3 = Up(64, 3)  # (64, 32, 32) -> (3, 64, 64)

    def forward(self, x, t):
        # t is ignored for now
        y_1, skip_1 = self.down_1(x, t)
        y_2, skip_2 = self.down_2(y_1, t)
        y_3, skip_3 = self.down_3(y_2, t)

        y_4 = self.bottleneck_1(y_3, t)
        y_5 = self.bottleneck_2(y_4, t)
        y_6 = self.bottleneck_3(y_5, t)

        y_7 = self.up_1(y_6, skip_3, t)  # (256, 16, 16) + (256, 16, 16)
        y_8 = self.up_2(y_7, skip_2, t)  # (128, 32, 32) + (128, 32, 32)
        y_9 = self.up_3(y_8, skip_1, t)  # (256, 16, 16) + (256, 16, 16)
        return y_9


model = UNet().to("cuda")
sample_input = torch.randn((1, 3, 64, 64)).to("cuda")
sample_output = model(sample_input, 5)
print(sample_output.shape)
plt.imshow(sample_output.cpu().detach().squeeze(0).permute(1, 2, 0))
plt.show()