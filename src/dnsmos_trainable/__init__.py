"""Trainable DNSMOS P.835 with ONNX exports (QDQ int8 forward, fused fwd+bwd loss graphs)."""

from dnsmos_trainable.constants import INPUT_LEN, SR

__all__ = ["INPUT_LEN", "SR", "load_transplanted"]


def load_transplanted(state_dict_path):
    """Build a :class:`DnsmosModel` initialized from a transplanted state dict."""
    import torch

    from dnsmos_trainable.model import DnsmosModel

    model = DnsmosModel()
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu", weights_only=True))
    model.eval()
    return model
