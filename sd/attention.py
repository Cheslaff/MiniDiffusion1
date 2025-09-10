import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads, n_embd, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(n_embd, 3 * n_embd, bias=in_proj_bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = n_embd // n_heads

    def forward(self, x, causal_mask=False):
        in_shape = x.shape
        B, T, C = in_shape
        interm_shape = (B, T, self.n_heads, self.d_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (B, NH, T, DH)
        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)

        # (B, NH, T, T)
        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).tril(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        out = weight @ v  # (B, NH, T, T) @ (B, NH, T, DH) -> (B, NH, T, DH)
        out = out.transpose(1, 2)
        out = out.reshape(in_shape)
        out = self.out_proj(out)

        return out
