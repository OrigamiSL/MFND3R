import torch


class OffDiagMask():
    def __init__(self, B, L, V, S, device="cpu"):
        _mask = torch.eye(L, S, dtype=torch.bool).to(device)
        self._mask = _mask.unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)

    @property
    def mask(self):
        return self._mask
