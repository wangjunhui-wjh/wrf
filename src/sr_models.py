import torch
import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(self, in_channels, out_channels, scale=4):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels * (scale ** 2), kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.shuffle(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, h, w = x.size()
        avg = torch.mean(x, dim=(2, 3), keepdim=False)
        mx = torch.amax(x, dim=(2, 3), keepdim=False)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * attn

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.amax(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_pool, max_pool], dim=1)
        spatial = torch.sigmoid(self.spatial(spatial))
        return x * spatial


class UNetSR(nn.Module):
    def __init__(self, in_channels, out_channels, base=32, scale=4, use_cbam=False):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.up1 = Up(base * 4, base * 2)
        self.up2 = Up(base * 2, base)
        self.cbam = CBAM(base * 4) if use_cbam else None
        self.outc = nn.Conv2d(base, out_channels * (scale ** 2), kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        if self.cbam is not None:
            x3 = self.cbam(x3)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        x = self.shuffle(x)
        return x


def build_model(model_name, in_channels, out_channels, scale=4):
    name = model_name.lower()
    if name == "espcn":
        return ESPCN(in_channels, out_channels, scale=scale)
    if name == "unet":
        return UNetSR(in_channels, out_channels, scale=scale, use_cbam=False)
    if name == "physr":
        return UNetSR(in_channels, out_channels, scale=scale, use_cbam=True)
    raise ValueError(f"Unknown model: {model_name}")
