"""
model.py
-----------------------
Lightweight NAFNet (Chen et al., "Simple Baselines for Image Restoration",
ECCV 2022) adapted for this challenge.

Input  (7 channels): noisy RGB | classically pre-cleaned RGB | noise sigma map
Output (3 channels): denoised RGB = pre-cleaned RGB + predicted residual

NAFNet is chosen because it has no self-attention and no non-linear
activations (SimpleGate instead), which makes it fast on CPU while giving
state-of-the-art denoising quality per FLOP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        # Simplified channel attention: global context -> channel weights.
        # This gives every pixel access to image-wide statistics (e.g. the
        # global colour offset and noise level), which a local CNN cannot see.
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)

        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg(x)
        x = x * self.sca(x)
        y = inp + self.conv3(x) * self.beta

        x = self.sg(self.conv4(self.norm2(y)))
        return y + self.conv5(x) * self.gamma


class NAFNet(nn.Module):
    def __init__(self, in_ch: int = 7, out_ch: int = 3, width: int = 32,
                 enc_blocks=(1, 1, 2, 4), middle_blocks: int = 4,
                 dec_blocks=(1, 1, 1, 1)):
        super().__init__()
        self.config = dict(in_ch=in_ch, out_ch=out_ch, width=width,
                           enc_blocks=list(enc_blocks), middle_blocks=middle_blocks,
                           dec_blocks=list(dec_blocks))
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.ending = nn.Conv2d(width, out_ch, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        c = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blocks)])
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.pad_multiple = 2 ** len(enc_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 7, H, W) = [noisy, cleaned, sigma]. Returns (B, 3, H, W)."""
        _, _, h, w = x.shape
        m = self.pad_multiple
        ph, pw = (m - h % m) % m, (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        base = x[:, 3:6]

        feat = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)
        feat = self.middle(feat)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = dec(up(feat) + skip)

        out = base + self.ending(feat)
        return out[:, :, :h, :w]


PRESETS = {
    # 11.6M params, ~8 s per 992x992 image on an 8-core CPU (default)
    "small": dict(width=32, enc_blocks=(1, 1, 2, 4), middle_blocks=4, dec_blocks=(1, 1, 1, 1)),
    # 2.9M params, ~3.6 s per image on CPU (fastest)
    "tiny": dict(width=16, enc_blocks=(1, 1, 2, 4), middle_blocks=4, dec_blocks=(1, 1, 1, 1)),
    # 21.8M params, ~15 s per image on CPU (highest capacity)
    "base": dict(width=32, enc_blocks=(2, 2, 4, 8), middle_blocks=8, dec_blocks=(2, 2, 2, 2)),
}


def build_model(preset: str = "small", **overrides) -> NAFNet:
    cfg = dict(PRESETS[preset])
    cfg.update(overrides)
    return NAFNet(**cfg)
