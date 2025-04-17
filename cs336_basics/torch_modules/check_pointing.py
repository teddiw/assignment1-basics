import torch
import torch.nn as nn
import numpy as np
from einops import einsum, reduce, rearrange
from jaxtyping import Float, Int
from torch import Tensor, LongTensor
import typing
import os

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, 
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = {}
    obj['optimizer'] = optimizer.state_dict()
    obj['model'] = model.state_dict()
    obj['iteration'] = iteration
    torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    ) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']
