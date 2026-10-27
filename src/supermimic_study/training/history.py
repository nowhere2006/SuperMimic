"""The same differentiable term-major history operation in sim and imagination."""
from __future__ import annotations

import torch

TERM_WIDTHS = (29, 29, 3, 3, 29)


def initial_history(frame: torch.Tensor, length: int = 10) -> torch.Tensor:
    if frame.shape[-1] != 93 or length < 1:
        raise ValueError("history requires 93D frames and positive length")
    return torch.cat([x.unsqueeze(-2).expand(*x.shape[:-1], length, w).flatten(-2)
                      for x, w in zip(frame.split(TERM_WIDTHS, -1), TERM_WIDTHS)], -1)


def advance_history(history: torch.Tensor, frame: torch.Tensor, length: int = 10) -> torch.Tensor:
    if history.shape[-1] != length * 93 or frame.shape[-1] != 93:
        raise ValueError("term-major history/frame width mismatch")
    result = []
    for old, new, width in zip(history.split([w*length for w in TERM_WIDTHS], -1),
                               frame.split(TERM_WIDTHS, -1), TERM_WIDTHS):
        window = old.reshape(*old.shape[:-1], length, width)
        result.append(torch.cat((window[..., 1:, :], new.unsqueeze(-2)), -2).flatten(-2))
    return torch.cat(result, -1)


def add_sensor_noise(frame: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    """Table A.1 uniforms, sampled outside the world model; previous action is clean."""
    if not enabled:
        return frame
    amplitude = frame.new_tensor([.01]*29 + [.5]*29 + [.2]*3 + [.05]*3 + [0.]*29)
    return frame + (2*torch.rand_like(frame)-1)*amplitude
