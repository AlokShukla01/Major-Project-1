"""Small DCGAN (Radford et al. 2015) for synthetic polyp image generation at 64x64.

    python -m polypseg.gan --images synthetic --epochs 20          # demo on procedural data
    python -m polypseg.gan --images "/path/to/LD Polyp/TrainValid/Images" --epochs 50

Generated images are unlabeled. To use them for segmentation you would need masks
(e.g. a conditional GAN or pseudo-labels from a trained model); see CompleteIdea.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from . import config as C

Z_DIM = 100
IMG = 64


class Generator(nn.Module):
    def __init__(self, z=Z_DIM, g=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z, g * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(g * 8), nn.ReLU(True),
            nn.ConvTranspose2d(g * 8, g * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(g * 4), nn.ReLU(True),
            nn.ConvTranspose2d(g * 4, g * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(g * 2), nn.ReLU(True),
            nn.ConvTranspose2d(g * 2, g, 4, 2, 1, bias=False), nn.BatchNorm2d(g), nn.ReLU(True),
            nn.ConvTranspose2d(g, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(d, d * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(d * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(d * 2, d * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(d * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(d * 4, d * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(d * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(d * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def _init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def load_image_folder(folder, limit=2000):
    paths = sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:limit]
    imgs = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is not None:
            imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    if not imgs:
        raise FileNotFoundError(f"No images found under {folder}")
    return imgs


def _to_tensor(images):
    arr = np.stack([cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_AREA) for im in images]).astype(np.float32)
    arr = arr / 127.5 - 1.0
    return torch.from_numpy(arr.transpose(0, 3, 1, 2))


def train_gan(images, epochs=20, batch_size=64, lr=2e-4, device=None, progress_callback=None, seed=C.SEED):
    """Train a DCGAN on a list of RGB uint8 images. Returns (generator, history)."""
    torch.manual_seed(seed)
    device = torch.device(device) if device else C.get_device()
    data = _to_tensor(images)
    G, D = Generator().to(device), Discriminator().to(device)
    G.apply(_init)
    D.apply(_init)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    history = {"d_loss": [], "g_loss": []}
    n = len(data)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        d_sum = g_sum = 0.0
        steps = 0
        for i in range(0, n, batch_size):
            real = data[perm[i:i + batch_size]].to(device)
            b = real.size(0)
            z = torch.randn(b, Z_DIM, 1, 1, device=device)
            fake = G(z)

            opt_d.zero_grad()
            d_loss = bce(D(real), torch.full((b,), 0.9, device=device)) + bce(D(fake.detach()), torch.zeros(b, device=device))
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()
            g_loss = bce(D(fake), torch.ones(b, device=device))
            g_loss.backward()
            opt_g.step()

            d_sum, g_sum, steps = d_sum + d_loss.item(), g_sum + g_loss.item(), steps + 1
        history["d_loss"].append(d_sum / steps)
        history["g_loss"].append(g_sum / steps)
        print(f"  GAN epoch {epoch + 1}/{epochs}  D {history['d_loss'][-1]:.3f}  G {history['g_loss'][-1]:.3f}")
        if progress_callback:
            progress_callback(epoch + 1, epochs, history)
    return G.eval(), history


@torch.no_grad()
def sample(generator, n=16, seed=None, out_size=128):
    """Generate n RGB uint8 images."""
    device = next(generator.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    z = torch.randn(n, Z_DIM, 1, 1, generator=g).to(device)
    x = generator(z).cpu().numpy().transpose(0, 2, 3, 1)
    x = ((x + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return [cv2.resize(im, (out_size, out_size), interpolation=cv2.INTER_CUBIC) for im in x]


def save_generator(G, path, history=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": G.state_dict(), "history": history or {}}, path)


def load_generator(path, device=None):
    device = torch.device(device) if device else C.get_device()
    ck = torch.load(path, map_location="cpu", weights_only=True)
    G = Generator()
    G.load_state_dict(ck["state_dict"])
    return G.to(device).eval(), ck.get("history", {})


GAN_CHECKPOINT = C.CHECKPOINT_DIR / "gan" / "dcgan.pt"


def main(argv=None):
    p = argparse.ArgumentParser(description="Train a DCGAN on polyp images")
    p.add_argument("--images", required=True, help='"synthetic" or a folder of images')
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n", type=int, default=800, help="synthetic sample count")
    p.add_argument("--out", default=str(GAN_CHECKPOINT))
    p.add_argument("--samples", type=int, default=16, help="images to write to outputs/gan_samples")
    args = p.parse_args(argv)

    if args.images == "synthetic":
        from .synthetic import generate_arrays
        imgs, _, _, _ = generate_arrays(args.n, h=128, w=128)
        images = list(imgs)
    else:
        images = load_image_folder(args.images)
    G, hist = train_gan(images, epochs=args.epochs)
    save_generator(G, args.out, hist)
    out_dir = C.OUTPUT_DIR / "gan_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(sample(G, args.samples, seed=0)):
        cv2.imwrite(str(out_dir / f"sample_{i:02d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    print(f"Generator saved to {args.out}; samples in {out_dir}")


if __name__ == "__main__":
    main()
