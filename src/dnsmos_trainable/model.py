"""DNSMOS P.835 CNN body, polynomial MOS mapping, and the full model."""

import torch
from torch import nn

from dnsmos_trainable.constants import OFFICIAL_CONFIG, POLY_BAK, POLY_OVR, POLY_SIG
from dnsmos_trainable.frontend import Frontend


class DnsmosBody(nn.Module):
    """CNN over the [B, 1, 900, 161] log-power spectrogram -> raw [B, 3].

    Topology of the official sig_bak_ovr.onnx: 3x3 "same" convs with ReLU,
    2x2/2 max pools (floor odd dims: (900,161)->(450,80)->(225,40)->(112,20)),
    global max over spatial dims, then 64->128->64->3 dense head.
    """

    def __init__(self, cfg=OFFICIAL_CONFIG) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        prev = 1
        for k in range(7):
            setattr(self, f"conv{k + 1}", nn.Conv2d(prev, c[k], 3, padding=1))
            prev = c[k]
        self.pool = nn.MaxPool2d(2, 2)
        f1, f2 = cfg.fc
        self.fc1 = nn.Linear(c[6], f1)
        self.fc2 = nn.Linear(f1, f2)
        self.fc3 = nn.Linear(f2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        x = torch.relu(self.conv6(x))
        x = self.pool(x)
        x = torch.relu(self.conv7(x))
        x = torch.amax(x, dim=(2, 3))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PolyMapping(nn.Module):
    """Raw [B, 3] -> mapped MOS [B, 3] via the official per-score quadratics."""

    def __init__(self) -> None:
        super().__init__()
        coefs = torch.tensor([POLY_SIG, POLY_BAK, POLY_OVR])  # [3, 3] highest-first
        self.register_buffer("c2", coefs[:, 0])
        self.register_buffer("c1", coefs[:, 1])
        self.register_buffer("c0", coefs[:, 2])

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.c2 * raw * raw + self.c1 * raw + self.c0


class DnsmosModel(nn.Module):
    """Full model: waveform [B, 144160] -> (raw [B, 3], mos [B, 3])."""

    def __init__(self, cfg=OFFICIAL_CONFIG) -> None:
        super().__init__()
        self.cfg = cfg
        self.frontend = Frontend(cfg)
        self.body = DnsmosBody(cfg)
        self.poly = PolyMapping()

    def forward(self, wav: torch.Tensor):
        raw = self.body(self.frontend(wav))
        return raw, self.poly(raw)
