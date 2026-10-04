"""Noise2Noise: odszumiacz uczony na danych z tej samej sesji (PyTorch, GPU).

Dwie połówki stosu (zdjęcia parzyste i nieparzyste) widzą to samo niebo z niezależnym szumem.
Sieć uczy się przewidywać jedną połówkę z drugiej; szumu przewidzieć się nie da, więc w średniej
wychodzi czysty sygnał (Lehtinen i in. 2018). Nic nie pochodzi spoza tych zdjęć: model zna tylko
tę matrycę, to ISO i to niebo. Dane liniowe (po odjęciu tła), w jednostkach σ szumu tła.
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger("skyhunt")


def _unet(ch: int = 3, base: int = 32):
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))

    class UNet(nn.Module):
        """Mały U-Net (3 poziomy), wyjście = wejście + poprawka (sieć uczy się tylko szumu)."""

        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = block(ch, base), block(base, 2 * base), block(2 * base, 4 * base)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.d2, self.d1 = block(6 * base, 2 * base), block(3 * base, base)
            self.out = nn.Conv2d(base, ch, 1)

        def forward(self, x):
            import torch

            g = torch.asinh(x / 4.0) * 4.0               # wejście ściśnięte (jądro galaktyki, gwiazdy)
            e1 = self.e1(g)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.up(e3), e2], 1))
            d1 = self.d1(torch.cat([self.up(d2), e1], 1))
            return x + self.out(d1)

    return UNet()


def noise_sigma(img: np.ndarray, bgmask: np.ndarray) -> float:
    v = img[bgmask][::5]
    v = v[np.isfinite(v)]
    if not len(v):
        return float(np.nanstd(img)) or 1.0
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) or 1.0


def train(a: np.ndarray, b: np.ndarray, *, steps: int, patch: int, batch: int, lr: float, device: str,
          seed: int = 0):
    """a, b: połówki [H, W, 3] w jednostkach σ. Losowe wycinki, losowy kierunek a→b / b→a,
    obroty i odbicia. Strata L2 (nieobciążona dla szumu o średniej 0), ważona jasnością WEJŚCIA
    (gwiazdy i jądro nie dominują; waga nie zależy od szumu celu)."""
    import torch

    torch.manual_seed(seed)
    A = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1), np.float32)).to(device)
    B = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1), np.float32)).to(device)
    _, H, W = A.shape
    net = _unet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu")
    t0 = time.perf_counter()
    for step in range(steps):
        ys = torch.randint(0, H - patch, (batch,), generator=gen)
        xs = torch.randint(0, W - patch, (batch,), generator=gen)
        swap = torch.rand(batch, generator=gen) < 0.5
        inp, tgt = [], []
        for k in range(batch):
            pa = A[:, ys[k]:ys[k] + patch, xs[k]:xs[k] + patch]
            pb = B[:, ys[k]:ys[k] + patch, xs[k]:xs[k] + patch]
            if swap[k]:
                pa, pb = pb, pa
            r = int(torch.randint(0, 4, (1,), generator=gen))
            pa, pb = torch.rot90(pa, r, (1, 2)), torch.rot90(pb, r, (1, 2))
            inp.append(pa)
            tgt.append(pb)
        inp, tgt = torch.stack(inp), torch.stack(tgt)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device != "cpu"):
            pred = net(inp)
        w = 1.0 / (1.0 + inp.abs().mean(dim=1, keepdim=True) / 10.0)
        loss = (w * (pred.float() - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (step + 1) % max(steps // 5, 1) == 0:
            log.info("Noise2Noise: krok %d/%d, strata %.3f (%.0f s)", step + 1, steps, float(loss),
                     time.perf_counter() - t0)
    return net.eval()


def apply(net, img: np.ndarray, device: str, tile: int = 512, overlap: int = 48) -> np.ndarray:
    """Odszumienie całego obrazu [H, W, 3] kafelkami z zakładką (łączenie wagą trójkątną)."""
    import torch

    H, W, _ = img.shape
    out = np.zeros_like(img, np.float32)
    acc = np.zeros((H, W, 1), np.float32)
    step = tile - 2 * overlap
    ramp = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).astype(np.float32)
    wtile = np.minimum(np.minimum.outer(ramp, ramp), overlap + 1)[..., None]
    with torch.no_grad():
        for y in range(0, max(H - 2 * overlap, 1), step):
            for x in range(0, max(W - 2 * overlap, 1), step):
                y0, x0 = min(y, max(H - tile, 0)), min(x, max(W - tile, 0))
                t = img[y0:y0 + tile, x0:x0 + tile]
                h, w = t.shape[:2]
                ph, pw = (-h) % 4, (-w) % 4               # U-Net: wymiary podzielne przez 4
                tt = np.pad(t, ((0, ph), (0, pw), (0, 0)), mode="reflect")
                inp = torch.from_numpy(np.ascontiguousarray(tt.transpose(2, 0, 1)))[None].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device != "cpu"):
                    pred = net(inp)
                p = pred.float()[0].cpu().numpy().transpose(1, 2, 0)[:h, :w]
                wt = wtile[:h, :w]
                out[y0:y0 + h, x0:x0 + w] += p * wt
                acc[y0:y0 + h, x0:x0 + w] += wt
    return out / np.maximum(acc, 1e-6)


def denoise_halves(a: np.ndarray, b: np.ndarray, bgmask: np.ndarray, cfg: dict, device: str) -> tuple[np.ndarray, dict]:
    """Uczenie na połówkach i odszumienie obu; wynik = średnia odszumionych połówek (każda ma
    poziom szumu taki, na jakim uczono sieć)."""
    sig = float(np.sqrt(0.5 * (noise_sigma(a.mean(axis=-1), bgmask) ** 2 + noise_sigma(b.mean(axis=-1), bgmask) ** 2)))
    an, bn = np.nan_to_num(a / sig), np.nan_to_num(b / sig)
    t0 = time.perf_counter()
    net = train(an, bn, steps=int(cfg["n2n_steps"]), patch=int(cfg["n2n_patch"]), batch=int(cfg["n2n_batch"]),
                lr=float(cfg["n2n_lr"]), device=device)
    den = 0.5 * (apply(net, an, device) + apply(net, bn, device)) * sig
    before = noise_sigma(0.5 * (a + b).mean(axis=-1), bgmask)
    after = noise_sigma(den.mean(axis=-1), bgmask)
    info = {"sigma_half": sig, "noise_before": before, "noise_after": after,
            "gain": before / max(after, 1e-9), "seconds": time.perf_counter() - t0}
    log.info("Noise2Noise: szum tła %.3g → %.3g (×%.1f mniej), %.0f s", before, after, info["gain"],
             info["seconds"])
    return den.astype(np.float32), info

