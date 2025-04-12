import torch
import torch.nn as nn
import numpy as np
from einops import einsum, reduce, rearrange
from jaxtyping import Float, Int
from torch import Tensor, LongTensor

class Linear(nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 weights: Float[Tensor, " d_out d_in"] | None = None, # for testing purposes
                 ):
        
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        if (weights != None):
            self.weights = nn.Parameter(weights, requires_grad=True)
        else:
            weights = torch.zeros([out_features, in_features])
            sigma = np.sqrt(2/(in_features + out_features))
            initialized_weights = torch.nn.init.trunc_normal_(weights, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
            self.weights = nn.Parameter(initialized_weights, requires_grad=True)

    def forward(self, 
                x: torch.Tensor,
                ) -> torch.Tensor:
        return einsum(x, self.weights, "batch seq d_in, d_out d_in-> batch seq d_out")

class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 weights: Float[Tensor, " vocab_size d_model"] | None = None, # for testing purposes
                 ):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # Initialize embedding weights
        if (weights != None):   
            self.embeddings = nn.Parameter(weights, requires_grad=True)
        else:
            embeddings = torch.zeros([num_embeddings, embedding_dim])
            initialized_embeddings = torch.nn.init.trunc_normal_(embeddings, mean=0, std=1, a=-3, b=3)
            self.embeddings = nn.Parameter(initialized_embeddings, requires_grad=True)
        
    def forward(self,
                token_ids: Int[LongTensor, "batch seq"],
                ) -> Float[Tensor, "batch seq d_model"]:
        return self.embeddings[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model:int,
                 eps:float=1e-5,
                 device:torch.device | None = None,
                 dtype:torch.dtype | None = None,
                 weights: Float[Tensor, " d_model"] | None = None, # for testing purposes
                 ):
        super(RMSNorm, self).__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        if (weights != None):
            gains = nn.Parameter(weights, requires_grad=True)
        else:
            initialized_weights = torch.ones(d_model)
            gains = nn.Parameter(initialized_weights, requires_grad=True)
        
        self.gains = rearrange(gains, 'd_model -> 1 1 d_model')

    def forward(self, 
                x: Float[Tensor, "batch seq d_model"],
                ) -> Float[Tensor, "batch seq d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)   
        rms = torch.sqrt(reduce(x**2, 'batch seq d_model -> batch seq', 'mean')+ self.eps)
        result = x * self.gains / rearrange(rms, 'b s -> b s 1')
        return result.to(in_dtype)
        