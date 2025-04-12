import torch
import torch.nn as nn
import numpy as np
from einops import einsum

class Linear(nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 weights:  Float[Tensor, " d_out d_in"] | None = None
                 ):
        
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        if (weights):
            self.weights = weights
        else:
            weights = torch.zeros([in_features,out_features])
            sigma = np.sqrt(2/(in_features + out_features))
            initialized_weights = torch.nn.init.trunc_normal_(weights, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
            self.weights = nn.Parameter(initialized_weights, requires_grad=True)

    def forward(self, 
                x: torch.Tensor,
                ) -> torch.Tensor:
        return einsum(x, self.weights, "batch sequence d_in, in d_out -> batch sequence d_out")

