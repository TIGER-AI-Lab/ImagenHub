import gc
from copy import deepcopy
from typing import Union

import torch
from torch import nn as nn
from torch.nn import functional as F


@torch.compile(fullgraph=True)
def fused_rms_norm(x: torch.Tensor, weight: nn.Parameter, eps: float):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps))) * weight


@torch.compile(fullgraph=True)
def fused_ada_layer_norm(C: int, eps: float, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    x = x.float()
    x = F.layer_norm(input=x, normalized_shape=(C,), weight=None, bias=None, eps=eps)
    return x.mul(scale.add(1)).add_(shift)


@torch.compile(fullgraph=True)
def fused_ada_rms_norm(C: int, eps: float, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    x = x.float()
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps)))
    return x.mul(scale.add(1)).add_(shift)
