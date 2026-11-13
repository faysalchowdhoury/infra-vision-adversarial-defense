"""Classifier, corruptions, attacks, input-transformation defences and the integrity score.

Classifier: a small CNN (3 conv blocks) on 32x32 crops, trained with light augmentation.

Threats (applied to test images):

* ``noise``: camera sensor noise, Gaussian sd 0.08;
* ``dirty_lens``: blur, haze (contrast loss) and random dark smudges;
* ``stickers``: two random black/white rectangles covering about 8% of the sign (graffiti, tape);
* ``adv_patch``: one *universal* 7x7-pixel patch (about 5% of the image) optimised against the model
  over random positions (expectation over transformation), then printed at a random spot on every sign;
* ``pgd4``, ``pgd8``: white-box L-infinity PGD at epsilon 4/255 and 8/255 (digital, upper bound);
* ``pgd8_adaptive``: the same PGD, but differentiating through the defence with BPDA (the defence is
  treated as identity on the backward pass), i.e. an attacker who knows the defence.

Defences (preprocessing before the classifier): 3x3 median filter, JPEG re-encoding at quality 50, and a
Fourier low-pass. **Integrity score**: feature squeezing (Xu et al., NDSS 2018), the L1 distance between
the softmax on the raw image and on a squeezed copy (4-bit colour + median); a high score flags an input
whose prediction is fragile, and the threshold is set for 5% false alarms on clean validation images.
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

N_CLASSES = 43


class SignNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, N_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0


def train(x: np.ndarray, y: np.ndarray, *, epochs: int = 8, seed: int = 0, batch: int = 128) -> SignNet:
    torch.manual_seed(seed)
    model = SignNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * int(np.ceil(len(x) / batch)))
    xt, yt = to_tensor(x), torch.tensor(y)
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(xt)).split(batch):
            xb = xt[b]
            # light augmentation: brightness/contrast jitter and small translations
            xb = (xb * (0.7 + 0.6 * torch.rand(len(b), 1, 1, 1)) + 0.1 * (torch.rand(len(b), 1, 1, 1) - 0.5)).clamp(0, 1)
            xb = torch.roll(xb, shifts=(int(torch.randint(-2, 3, ())), int(torch.randint(-2, 3, ()))), dims=(2, 3))
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yt[b])
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model: SignNet, x: torch.Tensor, batch: int = 512) -> torch.Tensor:
    return torch.cat([F.softmax(model(b), 1) for b in x.split(batch)])


# ----------------------------------------------------------------- threats
def noise(x: torch.Tensor, rng: np.random.Generator, sd: float = 0.08) -> torch.Tensor:
    return (x + torch.tensor(rng.normal(0, sd, x.shape), dtype=torch.float32)).clamp(0, 1)


def dirty_lens(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    k = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16
    out = F.conv2d(x, k.expand(3, 1, 3, 3), padding=1, groups=3)
    out = 0.6 * out + 0.4 * out.mean(dim=(1, 2, 3), keepdim=True)  # haze: contrast loss
    yy, xx = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
    for i in range(len(out)):
        for _ in range(3):
            cy, cx, r = rng.uniform(0, 32), rng.uniform(0, 32), rng.uniform(2, 5)
            m = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2))
            out[i] = out[i] * (1 - 0.6 * m)
    return out.clamp(0, 1)


def stickers(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    out = x.clone()
    for i in range(len(out)):
        for _ in range(2):
            h, w = int(rng.integers(3, 6)), int(rng.integers(5, 9))
            r, c = int(rng.integers(6, 26 - h)), int(rng.integers(4, 28 - w))
            out[i, :, r:r + h, c:c + w] = float(rng.choice([0.0, 1.0]))
    return out


def place_patch(x: torch.Tensor, patch: torch.Tensor, rows: np.ndarray, cols: np.ndarray) -> torch.Tensor:
    out = x.clone()
    p = patch.shape[-1]
    for i in range(len(out)):
        out[i, :, rows[i]:rows[i] + p, cols[i]:cols[i] + p] = patch
    return out


def train_patch(model: SignNet, x: torch.Tensor, y: torch.Tensor, *, size: int = 7, steps: int = 300, seed: int = 0) -> torch.Tensor:
    """Universal untargeted patch: maximise the loss over random images and positions (EOT)."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    patch = torch.rand(3, size, size, generator=g).requires_grad_(True)
    opt = torch.optim.Adam([patch], lr=0.05)
    for p in model.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (64,), generator=g)
        rows, cols = rng.integers(4, 28 - size, 64), rng.integers(4, 28 - size, 64)
        xb = x[idx].clone()
        for i in range(64):
            xb[i, :, rows[i]:rows[i] + size, cols[i]:cols[i] + size] = patch
        loss = -F.cross_entropy(model(xb), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            patch.clamp_(0, 1)
    for p in model.parameters():
        p.requires_grad_(True)
    return patch.detach()


def pgd(model: SignNet, x: torch.Tensor, y: torch.Tensor, eps: float, *, steps: int = 10, through=None) -> torch.Tensor:
    """L-infinity PGD. ``through`` (a defence) is applied on the forward pass with a straight-through
    (BPDA) gradient, which is how an attacker who knows the defence attacks it."""
    alpha = 2.5 * eps / steps
    out = []
    for xb, yb in zip(x.split(256), y.split(256)):
        adv = xb.clone()
        for _ in range(steps):
            adv.requires_grad_(True)
            inp = adv if through is None else adv + (through(adv.detach()) - adv).detach()
            loss = F.cross_entropy(model(inp), yb)
            (grad,) = torch.autograd.grad(loss, adv)
            adv = (adv.detach() + alpha * grad.sign()).clamp(xb - eps, xb + eps).clamp(0, 1)
        out.append(adv.detach())
    return torch.cat(out)


# ---------------------------------------------------------------- defences
def median3(x: torch.Tensor) -> torch.Tensor:
    p = F.pad(x, (1, 1, 1, 1), mode="reflect")
    return p.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*x.shape, 9).median(-1).values


def jpeg(x: torch.Tensor, quality: int = 50) -> torch.Tensor:
    out = []
    for img in (x.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        out.append(np.asarray(Image.open(io.BytesIO(buf.getvalue())), np.uint8))
    return to_tensor(np.stack(out))


def lowpass(x: torch.Tensor, keep: int = 10) -> torch.Tensor:
    f = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    c = x.shape[-1] // 2
    mask = torch.zeros_like(f.real)
    mask[..., c - keep:c + keep, c - keep:c + keep] = 1
    return torch.fft.ifft2(torch.fft.ifftshift(f * mask, dim=(-2, -1))).real.clamp(0, 1)


def squeeze(x: torch.Tensor) -> torch.Tensor:
    return median3(torch.round(x * 15) / 15)


DEFENCES = {"none": lambda x: x, "median": median3, "jpeg": jpeg, "lowpass": lowpass}


def integrity_score(model: SignNet, x: torch.Tensor) -> torch.Tensor:
    return (predict(model, x) - predict(model, squeeze(x))).abs().sum(1)
