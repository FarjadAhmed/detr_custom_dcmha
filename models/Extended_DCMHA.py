import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class Extended_DCMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(Extended_DCMHA, self).__init__()
        self.d_model = embed_dim
        self.nhead = num_heads
        self.rank = 64

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, **kwargs)
        self.tanh = nn.Tanh()
        
        # Learnable matrices to modify attention output
        self.Wb = nn.Parameter(torch.randn(embed_dim, embed_dim))
        # gating
        self.W_3g = nn.Parameter(torch.randn(embed_dim, embed_dim))

        self.W_K1 = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_k12 = nn.Parameter(torch.randn(int(embed_dim/2), self.rank))
        self.W_k12_upscale = nn.Parameter(torch.randn(self.rank, embed_dim))
        self.W_kg = nn.Parameter(torch.randn(int(embed_dim/2), self.rank))
        self.W_kg_mid = nn.Parameter(torch.randn(self.rank, embed_dim))
        # gating logic
        self.W_1g = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_2g = nn.Parameter(torch.randn(embed_dim, embed_dim))

        self.W_Q1 = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_q12 = nn.Parameter(torch.randn(int(embed_dim/2), self.rank))
        self.W_q12_upscale = nn.Parameter(torch.randn(self.rank, embed_dim))
        self.W_qg = nn.Parameter(torch.randn(int(embed_dim/2), self.rank))
        self.W_qg_mid = nn.Parameter(torch.randn(self.rank, embed_dim))
        # gating
        self.W_4g = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_5g = nn.Parameter(torch.randn(embed_dim, embed_dim))

        self._reset_parameters() # xavier initialization



    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.Wb)
        nn.init.xavier_uniform_(self.W_3g)
        nn.init.xavier_uniform_(self.W_K1)
        nn.init.xavier_uniform_(self.W_k12)
        nn.init.xavier_uniform_(self.W_k12_upscale)
        nn.init.xavier_uniform_(self.W_kg)
        nn.init.xavier_uniform_(self.W_kg_mid)
        nn.init.xavier_uniform_(self.W_1g)
        nn.init.xavier_uniform_(self.W_2g)
        nn.init.xavier_uniform_(self.W_Q1)
        nn.init.xavier_uniform_(self.W_q12)
        nn.init.xavier_uniform_(self.W_q12_upscale)
        nn.init.xavier_uniform_(self.W_qg)
        nn.init.xavier_uniform_(self.W_qg_mid)
        nn.init.xavier_uniform_(self.W_4g)
        nn.init.xavier_uniform_(self.W_5g)




    def forward(self, query, key, value, **kwargs):
        attn_output, attn_output_weights = self.mha(query, key, value, **kwargs)
        
        # Doing computations as per the figure 2(b) in the paper
        # Consider the branches are from 1-5 from left to right

        # branches left
        bleft = self.left_branch(attn_output, key)

        # branches right
        bright = self.right_branch(attn_output, query)

        # branch 3
        AWb = torch.matmul(attn_output, self.Wb)
        gated_AWb = self.tanh(torch.matmul(AWb, self.W_3g))

        output = bleft[0] + bleft[1] + gated_AWb + bright[0] + bright[1]
        
        return output, attn_output_weights

    def rmsnorm(self, tensor, dim):
        mean = tensor.mean(dim=dim, keepdim=True)
        var = tensor.var(dim=dim, keepdim=True, unbiased=False)
        rms = torch.sqrt(var + 1e-8)
        normed_tensor = tensor / rms
        return normed_tensor
    
    def left_branch(self, attn_output, key):
        # branch 1
        key_gelu = F.gelu(torch.matmul(key, self.W_K1))
        key_normed = self.rmsnorm(key_gelu, 0) # rmsnorm applied on along sequence
        key_chunked = torch.chunk(key_normed, chunks=2, dim=-1)
        b1 = torch.matmul(key_chunked[0], self.W_kg)
        b1 = torch.matmul(b1, self.W_kg_mid)
        b1 = (b1*attn_output)
        gated_b1 = self.tanh(torch.matmul(b1, self.W_1g))

        # branch 2
        b2 = torch.matmul(key_chunked[1], self.W_k12)
        b2 = torch.matmul(b2, self.W_k12_upscale)
        gated_b2 = self.tanh(torch.matmul(b2, self.W_2g))
        return gated_b1, gated_b2
    
    def right_branch(self, attn_output, query):
        # branch 5
        query_gelu = F.gelu(torch.matmul(query, self.W_Q1))
        query_normed = self.rmsnorm(query_gelu, 0) # rmsnorm applied on along sequence
        query_chunked = torch.chunk(query_normed, chunks=2, dim=-1)
        b1 = torch.matmul(query_chunked[0], self.W_qg)
        b1 = torch.matmul(b1, self.W_qg_mid)
        b1 = (b1*attn_output)
        gated_b1 = self.tanh(torch.matmul(b1, self.W_5g))

        # branch 4
        b2 = torch.matmul(query_chunked[1], self.W_q12)
        b2 = torch.matmul(b2, self.W_q12_upscale)
        gated_b2 = self.tanh(torch.matmul(b2, self.W_4g))
        return gated_b1, gated_b2