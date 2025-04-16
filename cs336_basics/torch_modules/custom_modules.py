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
            weights = torch.zeros([out_features, in_features], device=device, dtype=dtype)
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
            embeddings = torch.zeros([num_embeddings, embedding_dim], device=device, dtype=dtype)
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
            initialized_weights = torch.ones(d_model, device=device, dtype=dtype)
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
    
class PositionwiseFeedforward(nn.Module):
    def __init__(self,
                 d_model: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 d_ff: int = 0, # for testing purposes
                 w1_weight: Float[Tensor, " d_ff d_model"] | None = None, # for testing purposes
                 w2_weight: Float[Tensor, " d_model d_ff"] | None = None, # for testing purposes
                 w3_weight: Float[Tensor, " d_ff d_model"] | None = None, # for testing purposes
                ): 
        super(PositionwiseFeedforward, self).__init__()
        self.d_model = d_model
        if (d_ff):
            self.d_ff = d_ff
        else:
            self.d_ff = int(((d_model * 8/3) // 64) * 64) # multiple of 64
        self.device = device
        self.dtype = dtype

        if (w1_weight != None):
            # assume d_ff, w2_weight, and w3_weight are also provided
            self.w1 = nn.Parameter(w1_weight, requires_grad=True)
            self.w2 = nn.Parameter(w2_weight, requires_grad=True)   
            self.w3 = nn.Parameter(w3_weight, requires_grad=True)
        else:
            # initialize weights
            weights1 = torch.zeros([self.d_ff, d_model], device=device, dtype=dtype)
            weights2 = torch.zeros([d_model, self.d_ff], device=device, dtype=dtype)
            weights3 = torch.zeros([self.d_ff, d_model], device=device, dtype=dtype)
            sigma = np.sqrt(2/(self.d_ff, d_model))
            initialized_w1 = torch.nn.init.trunc_normal_(weights1, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
            initialized_w2 = torch.nn.init.trunc_normal_(weights2, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
            initialized_w3 = torch.nn.init.trunc_normal_(weights3, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
            self.w1 = nn.Parameter(initialized_w1, requires_grad=True)
            self.w2 = nn.Parameter(initialized_w2, requires_grad=True)
            self.w3 = nn.Parameter(initialized_w3, requires_grad=True)
    
    def forward(self,
                x: Float[Tensor, "batch seq d_model"],
                ) -> Float[Tensor, "batch seq d_model"]:
        temp1 = einsum(x, self.w1, "batch seq d_model, d_ff d_model -> batch seq d_ff")
        temp2 = torch.sigmoid(temp1) * temp1
        temp3 = einsum(x, self.w3, "batch seq d_model, d_ff  d_model -> batch seq d_ff")
        temp4 = temp2 * temp3
        result = einsum(temp4, self.w2, "batch seq d_ff, d_model  d_ff -> batch seq d_model")
        return result
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Create the positional encodings
        assert self.d_k % 2 == 0
        d_2 = self.d_k // 2
        precomputed_rotation_matrices = torch.zeros([d_2*self.max_seq_len, 4], device=device)
        for i in range(self.max_seq_len):
            for k in range(1, d_2+1):
                theta_k = i / (self.theta**(2*(k-1)/self.d_k))
                precomputed_rotation_matrices[i*d_2+k-1, 0] = np.cos(theta_k)
                precomputed_rotation_matrices[i*d_2+k-1, 1] = -np.sin(theta_k)
                precomputed_rotation_matrices[i*d_2+k-1, 2] = np.sin(theta_k)
                precomputed_rotation_matrices[i*d_2+k-1, 3] = np.cos(theta_k)
        self.register_buffer('precomputed_rotation_matrices', precomputed_rotation_matrices, persistent=True)

    def forward(self,
                x: Float[Tensor, "... seq d_k"],
                token_positions: Int[LongTensor, "... seq"]=None,
                ) -> Float[Tensor, "... seq d_k"]:
        
        if (token_positions == None):
            token_positions = np.arange(x.size(-2))
        block_vector = rearrange(self.precomputed_rotation_matrices, '(seq d_k) (rot1 rot2) -> seq d_k rot1 rot2', seq=self.max_seq_len, rot1=2)
        
        rotation_matrices = torch.stack([torch.block_diag(*block_vector[i]) for i in range(self.max_seq_len)])
        
        rotation_matrices_sliced = rotation_matrices[token_positions, :, :]
        # rotation_matrices has shape (max_seq_len, d_k, d_k)
        result = einsum(rotation_matrices_sliced, x, "... seq d_k1 d_k2, ... seq d_k2 -> ... seq d_k1")
        return result

def scaled_dot_product_attention(Q: Float[Tensor, "... n_queries d_k"],
                                 K: Float[Tensor, "... m_keys d_k"],
                                 V: Float[Tensor, "... m_keys d_v"],
                                 mask: Float[Tensor, "... n_queries m_keys"] | None = None, # " ... seq seq"
                                 ) -> Float[Tensor, "... n_queries d_v"]:

    temp1 = einsum(Q, K, "... n_queries d_k, ... m_keys d_k -> ... n_queries m_keys") / np.sqrt(Q.shape[-1])

    if mask is not None:   
        temp1 = temp1.masked_fill(mask == 0, -np.inf)

    temp2 = softmax(temp1, dim=-1)

    output = einsum(temp2, V, "... n_queries m_keys, ... m_keys d_v -> ... n_queries d_v")
    return output

class MultiheadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 max_seq_len: int | None = None,
                 theta: float | None = None,
                 token_positions: Int[LongTensor, "... seq"] | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 q_proj_weight: Float[Tensor, " d_k d_in"] | None = None, # for testing purposes
                 k_proj_weight: Float[Tensor, " d_k d_in"] | None = None, # for testing purposes
                 v_proj_weight: Float[Tensor, " d_v d_in"] | None = None, # for testing purposes
                 o_proj_weight: Float[Tensor, " d_model d_v"] | None = None, # for testing purposes
                 ):
        super(MultiheadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.token_positions = token_positions
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        if (q_proj_weight != None):
            # assume k_proj_weight, v_proj_weight, and o_proj_weight are also provided
            self.W_q = nn.Parameter(q_proj_weight, requires_grad=True)
            self.W_k = nn.Parameter(k_proj_weight, requires_grad=True)
            self.W_v = nn.Parameter(v_proj_weight, requires_grad=True)
            self.W_o = nn.Parameter(o_proj_weight, requires_grad=True)
        else:
            # initialize weights. These are all square matrices of size d_model.
            weights_q = torch.zeros([self.d_k*num_heads, d_model], device=device, dtype=dtype)
            weights_k = torch.zeros([self.d_k*num_heads, d_model], device=device, dtype=dtype)
            weights_v = torch.zeros([self.d_k*num_heads, d_model], device=device, dtype=dtype)
            weights_o = torch.zeros([d_model, self.d_k*num_heads], device=device, dtype=dtype)
            sigma = np.sqrt(2/(self.d_k*num_heads, d_model))
            initialized_wq = torch.nn.init.trunc_normal_(weights_q, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
            initialized_wk = torch.nn.init.trunc_normal_(weights_k, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
            initialized_wv = torch.nn.init.trunc_normal_(weights_v, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
            initialized_wo = torch.nn.init.trunc_normal_(weights_o, mean=0, std=sigma, a=-3*sigma, b=3*sigma)

            self.W_q = nn.Parameter(initialized_wq, requires_grad=True)
            self.W_k = nn.Parameter(initialized_wk, requires_grad=True)
            self.W_v = nn.Parameter(initialized_wv, requires_grad=True)
            self.W_o = nn.Parameter(initialized_wo, requires_grad=True)

        if (theta == None):
            self.rope = None
        else:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)

    def forward(self,
                x: Float[Tensor, " ... batch seq d_model"],
                ) -> Float[Tensor, " ... batch seq d_model"]: # d_model = d_k * num_heads
        # get Q, K, V
        Q = einsum(x, self.W_q, "... batch seq d_model, h_d_k d_model -> ... batch seq h_d_k")
        K = einsum(x, self.W_k, "... batch seq d_model, h_d_k d_model -> ... batch seq h_d_k")
        V = einsum(x, self.W_v, "... batch seq d_model, h_d_k d_model -> ... batch seq h_d_k")

        # Rearrange Q, K, V to have the head dimension as a batch dimension
        Q = rearrange(Q, "... batch seq (h d_k) -> ... batch h seq d_k", h = self.num_heads)
        K = rearrange(K, "... batch seq (h d_k) -> ... batch h seq d_k", h = self.num_heads)
        V = rearrange(V, "... batch seq (h d_v) -> ... batch h seq d_v", h = self.num_heads)

        # RoPE 
        if (self.rope):
            Q = self.rope.forward(Q, self.token_positions)
            K = self.rope.forward(K, self.token_positions)
        
        seq = x.shape[-2]
        causal_attn_mask = torch.tril(torch.ones([seq, seq], device=self.device, dtype=self.dtype))
        result = scaled_dot_product_attention(Q, K, V, mask=causal_attn_mask)

        result = rearrange(result, "... batch h seq d_v -> ... batch seq (h d_v)")
        # result = einsum(self.W_o, result, "d_model h_d_k, ... batch seq d_model -> ... batch seq d_model")
        result = einsum(result, self.W_o, "... batch seq h_d_k, d_model h_d_k-> ... batch seq d_model")

        return result

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 eps: float = 1e-5,
                 token_positions: Int[LongTensor, "... seq"] | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 weights: dict[str, Tensor] | None = None, # for testing purposes
                 ):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        if (weights): # if testing
            # unpack the weights from the dictionary
            rms_norm1_weights =  weights['ln1.weight']
            k_proj_weight = weights['attn.k_proj.weight']
            q_proj_weight = weights['attn.q_proj.weight']
            v_proj_weight = weights['attn.v_proj.weight']
            o_proj_weight = weights['attn.output_proj.weight']
            rms_norm2_weights = weights['ln2.weight']
            ffn_w1_weight =  weights['ffn.w1.weight']
            ffn_w2_weight =  weights['ffn.w2.weight']
            ffn_w3_weight =  weights['ffn.w3.weight']
        else:
            # set the weights to None
            rms_norm1_weights = None
            k_proj_weight = None
            q_proj_weight = None
            v_proj_weight = None
            o_proj_weight = None
            rms_norm2_weights = None
            ffn_w1_weight =  None
            ffn_w2_weight =  None
            ffn_w3_weight =  None

        self.rms_norm1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype, weights=rms_norm1_weights)
        self.multihead_attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, token_positions=token_positions, device=device, dtype=dtype, k_proj_weight=k_proj_weight, q_proj_weight=q_proj_weight, v_proj_weight=v_proj_weight, o_proj_weight=o_proj_weight)
        self.rms_norm2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype, weights=rms_norm2_weights)
        self.ffn = PositionwiseFeedforward(d_model, device=device, dtype=dtype, w1_weight=ffn_w1_weight, w2_weight=ffn_w2_weight, w3_weight=ffn_w3_weight)

    def forward(self, 
                x: Float[Tensor, " ... batch seq d_model"],
                ) -> Float[Tensor, " ... batch seq d_model"]:
        step1 = self.rms_norm1.forward(x)
        step2 = self.multihead_attn.forward(step1)
        step3 = x + step2
        step4 = self.rms_norm2.forward(step3)
        step5 = self.ffn.forward(step4)
        return step5 + step3
    
class TransformerLM(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 context_length: int,
                 theta: float,
                 vocab_size: int,
                 num_layers: int,
                 eps: float = 1e-5,
                 token_positions: Int[LongTensor, "... seq"] | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 weights: dict[str, Tensor] | None = None, # for testing purposes
                 ):
        super(TransformerLM, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        if (weights): # if testing
            embedding_weights =  weights['token_embeddings.weight']
            transformer_weights = []
            for i in range(num_layers):
                curr_weights = {}
                curr_weights[f'attn.q_proj.weight'] = weights[f'layers.{i}.attn.q_proj.weight']
                curr_weights[f'attn.k_proj.weight'] = weights[f'layers.{i}.attn.k_proj.weight']
                curr_weights[f'attn.v_proj.weight'] = weights[f'layers.{i}.attn.v_proj.weight']
                curr_weights[f'attn.output_proj.weight'] = weights[f'layers.{i}.attn.output_proj.weight']
                curr_weights[f'ln1.weight'] = weights[f'layers.{i}.ln1.weight']
                curr_weights[f'ln2.weight'] = weights[f'layers.{i}.ln2.weight']
                curr_weights[f'ffn.w1.weight'] = weights[f'layers.{i}.ffn.w1.weight']
                curr_weights[f'ffn.w2.weight'] = weights[f'layers.{i}.ffn.w2.weight']
                curr_weights[f'ffn.w3.weight'] = weights[f'layers.{i}.ffn.w3.weight']
                transformer_weights.append(curr_weights)

            final_linear_weights = weights['ln_final.weight']
            head_weights = weights['lm_head.weight']


        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=context_length, device=device, dtype=dtype, weights=embedding_weights)
        self.transformer_blocks = []
        for i in range(num_layers):
            self.transformer_blocks.append(TransformerBlock(d_model, num_heads, d_ff, context_length, theta, eps=eps, token_positions=token_positions, device=device, dtype=dtype, weights=transformer_weights[i]))
        self.norm = RMSNorm(d_model, eps=eps, device=device, dtype=dtype, weights=final_linear_weights)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype, weights=head_weights)
    
    def forward(self,
                token_ids: Int[Tensor, "... batch seq"],
                ) -> Float[Tensor, "... batch seq vocab_size"]:
        x = self.embedding.forward(token_ids)
        for transformer_block in self.transformer_blocks:
            x = transformer_block.forward(x)
        x = self.norm.forward(x)
        x = self.linear.forward(x)
        return x

def softmax(x: Float[Tensor, "..."],
            dim: int,
            ) -> Float[Tensor, "..."]:
    x_max = torch.max(x, dim=dim, keepdim=True)
    x = x - x_max.values
    x_exp = torch.exp(x)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp/x_exp_sum

def cross_entropy_loss(logits: Float[Tensor, '... batch vocab_size'],
                       targets: Int[Tensor, '... batch']
                       ) -> Float[Tensor, '...']:
    x_max = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - x_max.values
    exp_logits_sum = reduce(torch.exp(logits), '... batch vocab_size -> batch', 'sum')
    log_exp_logits_sum = torch.log(exp_logits_sum)

    # get the logits for the target tokens
    # targets = rearrange(targets, '... batch -> ... batch 1')
    row_idxs = np.arange(logits.shape[-2])
    target_logits = logits[row_idxs, targets] # '... batch'
    losses_by_batch = log_exp_logits_sum - target_logits

    return torch.mean(losses_by_batch)
