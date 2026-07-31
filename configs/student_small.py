"""Compact DNSMOS student for STM32N6 internal-SRAM deployment.

The full-size model's peak activation (128 x 900 x 161 ~ 18.5 MB int8) needs
external PSRAM on the N6. This student strides the first conv and cuts
channels so the int8 activation peak stays under ~1 MB, fitting the ~2.5 MB
of fast internal RAM with room for buffers. It reuses the exact transplanted
frontend (frozen by default) and is trained with scripts/train_distill.py
against the official teacher; expect correlation with the teacher (gate:
Pearson r >= 0.9 on OVRL), not exact score parity.
"""

import torch
from torch import nn

from dnsmos_trainable.frontend import Frontend
from dnsmos_trainable.model import PolyMapping


class StudentSmallBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # (900,161)->(450,81); pools floor odd dims after
        self.conv2 = nn.Conv2d(16, 24, 3, padding=1)
        self.conv3 = nn.Conv2d(24, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))  # (225,40)
        x = self.pool(torch.relu(self.conv3(x)))  # (112,20)
        x = self.pool(torch.relu(self.conv4(x)))  # (56,10)
        x = torch.amax(x, dim=(2, 3))
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class StudentSmall(nn.Module):
    """Same (raw, mos) interface as DnsmosModel; ~21k body parameters."""

    def __init__(self, freeze_frontend: bool = True) -> None:
        super().__init__()
        self.frontend = Frontend()
        self.body = StudentSmallBody()
        self.poly = PolyMapping()
        if freeze_frontend:
            for p in self.frontend.parameters():
                p.requires_grad_(False)

    def load_frontend_from_transplant(self, state_dict: dict) -> None:
        self.frontend.stft.w_re.data.copy_(state_dict["frontend.stft.w_re"])
        self.frontend.stft.w_im.data.copy_(state_dict["frontend.stft.w_im"])

    def forward(self, wav: torch.Tensor):
        raw = self.body(self.frontend(wav))
        return raw, self.poly(raw)
