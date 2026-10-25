"""Segmentation architectures used in the comparison (copied from the notebook)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BASE_CHANNELS


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class BasicUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base=BASE_CHANNELS):
        super().__init__()
        self.inc = DoubleConv(n_channels, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv_up1 = DoubleConv(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv_up2 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.conv_up1(torch.cat([x2, self.up1(x3)], dim=1))
        x = self.conv_up2(torch.cat([x1, self.up2(x)], dim=1))
        return self.outc(x)


class AttentionBlock(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())

    def forward(self, gate, skip):
        attn = self.psi(self.gate(gate) + self.skip(skip))
        return skip * attn


class AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base=BASE_CHANNELS):
        super().__init__()
        self.inc = DoubleConv(n_channels, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att1 = AttentionBlock(base * 2, base * 2, base)
        self.conv_up1 = DoubleConv(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att2 = AttentionBlock(base, base, max(1, base // 2))
        self.conv_up2 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        u1 = self.up1(x3)
        x2_att = self.att1(u1, x2)
        x = self.conv_up1(torch.cat([x2_att, u1], dim=1))
        u2 = self.up2(x)
        x1_att = self.att2(u2, x1)
        x = self.conv_up2(torch.cat([x1_att, u2], dim=1))
        return self.outc(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base=BASE_CHANNELS):
        super().__init__()
        self.x00 = DoubleConv(n_channels, base)
        self.x10 = DoubleConv(base, base * 2)
        self.x20 = DoubleConv(base * 2, base * 4)
        self.x01 = DoubleConv(base + base, base)
        self.x11 = DoubleConv(base * 2 + base * 2, base * 2)
        self.x02 = DoubleConv(base + base + base, base)
        self.pool = nn.MaxPool2d(2)
        self.up10 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.up20 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.up11 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.outc = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x01 = self.x01(torch.cat([x00, self.up10(x10)], dim=1))
        x11 = self.x11(torch.cat([x10, self.up20(x20)], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self.up11(x11)], dim=1))
        return self.outc(x02)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=4, dilation=4), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
        ])
        self.project = nn.Sequential(nn.Conv2d(out_ch * 4, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base=BASE_CHANNELS):
        super().__init__()
        self.enc1 = DoubleConv(n_channels, base)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.aspp = ASPP(base * 4, base * 2)
        self.low_project = nn.Sequential(nn.Conv2d(base, base, 1), nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(DoubleConv(base * 3, base * 2), nn.Conv2d(base * 2, n_classes, 1))

    def forward(self, x):
        low = self.enc1(x)
        x = self.enc2(low)
        x = self.enc3(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([self.low_project(low), x], dim=1)
        return self.decoder(x)


MODEL_FACTORIES = {
    "Basic U-Net": BasicUNet,
    "Attention U-Net": AttentionUNet,
    "U-Net++": UNetPlusPlus,
    "DeepLabV3+": DeepLabV3Plus,
}

# Deepest feature block of each model, used as the default Grad-CAM target layer.
GRADCAM_LAYERS = {
    "Basic U-Net": "down2",
    "Attention U-Net": "down2",
    "U-Net++": "x20",
    "DeepLabV3+": "aspp",
}


def build_model(name: str, base: int = BASE_CHANNELS) -> nn.Module:
    if name not in MODEL_FACTORIES:
        raise KeyError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_FACTORIES)}")
    return MODEL_FACTORIES[name](base=base)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
