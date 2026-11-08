"""Segmentation architectures used in the comparison (copied from the notebook)."""

from pathlib import Path

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


class ResidualConv(nn.Module):
    """Two-convolution residual block used by the hybrid architecture."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (nn.Identity() if in_ch == out_ch else
                     nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        return F.relu(self.main(x) + self.skip(x), inplace=True)


class HybridPolypNet(nn.Module):
    """Residual attention U-Net with ASPP multi-scale context for polyp masks."""

    def __init__(self, n_channels=3, n_classes=1, base=BASE_CHANNELS):
        super().__init__()
        self.enc1 = ResidualConv(n_channels, base)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(base, base * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(base * 2, base * 4))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(base * 4, base * 8))
        self.context = ASPP(base * 8, base * 4)

        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.att3 = AttentionBlock(base * 4, base * 4, base * 2)
        self.dec3 = ResidualConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionBlock(base * 2, base * 2, base)
        self.dec2 = ResidualConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionBlock(base, base, max(1, base // 2))
        self.dec1 = ResidualConv(base * 2, base)
        self.outc = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.context(self.enc4(x3))

        d3 = self.up3(x4)
        d3 = self.dec3(torch.cat([self.att3(d3, x3), d3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self.att2(d2, x2), d2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self.att1(d1, x1), d1], dim=1))
        return self.outc(d1)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class PretrainedResNet18UNet(nn.Module):
    """U-Net decoder with an ImageNet-pretrained ResNet-18 encoder."""

    def __init__(self, n_channels=3, n_classes=1, base=BASE_CHANNELS):
        super().__init__()
        if n_channels != 3:
            raise ValueError("Pretrained ResNet18 U-Net expects RGB input.")
        try:
            from torchvision.models import resnet18
        except ImportError as exc:
            raise ImportError("Pretrained ResNet18 U-Net requires torchvision; install requirements.txt.") from exc

        self.encoder = resnet18(weights=None)
        weights_path = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
        if weights_path.is_file():
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state, strict=False)
            self.pretrained_encoder = True
        else:
            # Loading a project checkpoint replaces these weights in inference; during training,
            # the caller can identify this flag and warn that ImageNet weights were unavailable.
            self.pretrained_encoder = False

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("input_mean", self.mean)
        self.register_buffer("input_std", self.std)
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        self.decoder0 = DecoderBlock(64, 64, max(16, base // 2))
        self.head = nn.Sequential(
            nn.Conv2d(max(16, base // 2), max(16, base // 2), 3, padding=1),
            nn.BatchNorm2d(max(16, base // 2)), nn.ReLU(inplace=True),
            nn.Conv2d(max(16, base // 2), n_classes, 1),
        )

    def forward(self, x):
        output_size = x.shape[2:]
        x = (x - self.input_mean) / self.input_std
        e0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        e1 = self.encoder.layer1(self.encoder.maxpool(e0))
        e2 = self.encoder.layer2(e1)
        e3 = self.encoder.layer3(e2)
        e4 = self.encoder.layer4(e3)
        x = self.decoder3(e4, e3)
        x = self.decoder2(x, e2)
        x = self.decoder1(x, e1)
        x = self.decoder0(x, e0)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.head(x)


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatially aligned frame features."""

    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, 3, padding=1)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def init_state(self, x):
        shape = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        return x.new_zeros(shape), x.new_zeros(shape)


class TemporalAttentionUNet(nn.Module):
    """U-Net with a ConvLSTM + temporal attention bottleneck.

    Input shape is B,T,C,H,W; output is the segmentation logits for the centre frame.
    The encoder is shared across frames, preserving compute and parameter efficiency.
    """

    def __init__(self, n_channels=3, n_classes=1, base=BASE_CHANNELS, temporal_window=5):
        super().__init__()
        self.temporal_window = temporal_window
        self.enc1 = DoubleConv(n_channels, base)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 4, base * 8))
        self.temporal_cell = ConvLSTMCell(base * 8, base * 8)
        self.temporal_score = nn.Conv2d(base * 8, 1, kernel_size=1)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, n_classes, 1)

    def forward(self, frames):
        if frames.ndim != 5:
            raise ValueError("TemporalAttentionUNet expects input shaped (batch, time, channels, height, width).")
        batch, steps, channels, height, width = frames.shape
        centre = steps // 2
        flat = frames.reshape(batch * steps, channels, height, width)
        e1 = self.enc1(flat)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        deep = self.bottleneck(e3)
        deep = deep.reshape(batch, steps, *deep.shape[1:])

        state = self.temporal_cell.init_state(deep[:, 0])
        recurrent = []
        for t in range(steps):
            state = self.temporal_cell(deep[:, t], state)
            recurrent.append(state[0])
        recurrent = torch.stack(recurrent, dim=1)
        attention = torch.softmax(self.temporal_score(recurrent.flatten(0, 1)).reshape(
            batch, steps, 1, *recurrent.shape[-2:]), dim=1)
        context = (recurrent * attention).sum(dim=1)

        e1 = e1.reshape(batch, steps, *e1.shape[1:])[:, centre]
        e2 = e2.reshape(batch, steps, *e2.shape[1:])[:, centre]
        e3 = e3.reshape(batch, steps, *e3.shape[1:])[:, centre]
        x = self.dec3(torch.cat([e3, self.up3(context)], dim=1))
        x = self.dec2(torch.cat([e2, self.up2(x)], dim=1))
        x = self.dec1(torch.cat([e1, self.up1(x)], dim=1))
        return self.outc(x)


class PolypPVTModel(nn.Module):
    """Official polyp-specific PVT model with an input/output adapter for PolypSeg."""

    def __init__(self, n_channels=3, n_classes=1, base=BASE_CHANNELS):
        super().__init__()
        if n_channels != 3 or n_classes != 1:
            raise ValueError("Polyp-PVT expects RGB input and a single binary mask output.")
        try:
            from .vendor.polyp_pvt.pvt import PolypPVT
        except ImportError as exc:
            raise ImportError("Polyp-PVT requires the timm package; install requirements.txt.") from exc
        self.network = PolypPVT()
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        output_size = x.shape[-2:]
        x = (x - self.input_mean) / self.input_std
        pred_coarse, pred_refined = self.network(x)
        logits = pred_coarse + pred_refined
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


MODEL_FACTORIES = {
    "Basic U-Net": BasicUNet,
    "Attention U-Net": AttentionUNet,
    "U-Net++": UNetPlusPlus,
    "DeepLabV3+": DeepLabV3Plus,
    "Hybrid Polyp U-Net": HybridPolypNet,
    "Pretrained ResNet18 U-Net": PretrainedResNet18UNet,
    "Temporal Attention U-Net": TemporalAttentionUNet,
    "Polyp-PVT": PolypPVTModel,
}

# Deepest feature block of each model, used as the default Grad-CAM target layer.
GRADCAM_LAYERS = {
    "Basic U-Net": "down2",
    "Attention U-Net": "down2",
    "U-Net++": "x20",
    "DeepLabV3+": "aspp",
    "Hybrid Polyp U-Net": "context",
    "Pretrained ResNet18 U-Net": "encoder.layer4",
    "Temporal Attention U-Net": "bottleneck",
    # The backbone returns a list of pyramid tensors, which is not a valid
    # single Grad-CAM activation. CFM returns one spatial feature tensor.
    "Polyp-PVT": "network.CFM",
}


def build_model(name: str, base: int = BASE_CHANNELS) -> nn.Module:
    if name not in MODEL_FACTORIES:
        raise KeyError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_FACTORIES)}")
    return MODEL_FACTORIES[name](base=base)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
