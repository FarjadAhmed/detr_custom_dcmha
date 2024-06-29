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
        
        # Learnable matrices to modify attention output
        self.Wb = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        self.W_K1 = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_k12 = nn.Parameter(torch.randn(int(embed_dim/2), self.rank))
        self.W_k12_upscale = nn.Parameter(torch.randn(self.rank, embed_dim))
        self.W_kg = nn.Parameter(torch.randn(int(embed_dim/2), self.rank))
        self.W_kg_mid = nn.Parameter(torch.randn(self.rank, embed_dim))

        self.W_Q1 = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_q12 = nn.Parameter(torch.randn(int(embed_dim/2), self.rank))
        self.W_q12_upscale = nn.Parameter(torch.randn(self.rank, embed_dim))
        self.W_qg = nn.Parameter(torch.randn(int(embed_dim/2), self.rank))
        self.W_qg_mid = nn.Parameter(torch.randn(self.rank, embed_dim))

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


        # I think im either missing the gating logic or I need to go 
        # in pytorch funcitonal and apply this before and after the softmax
        # first thing makes more sense, second sounds stupid

        output = bleft[0] + bleft[1] + AWb + bright[0] + bright[1]
        
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

        # branch 2
        b2 = torch.matmul(key_chunked[1], self.W_k12)
        b2 = torch.matmul(b2, self.W_k12_upscale)
        return b1, b2
    
    def right_branch(self, attn_output, query):
        # branch 5
        query_gelu = F.gelu(torch.matmul(query, self.W_Q1))
        query_normed = self.rmsnorm(query_gelu, 0) # rmsnorm applied on along sequence
        query_chunked = torch.chunk(query_normed, chunks=2, dim=-1)
        b1 = torch.matmul(query_chunked[0], self.W_qg)
        b1 = torch.matmul(b1, self.W_qg_mid)
        b1 = (b1*attn_output)

        # branch 4
        b2 = torch.matmul(query_chunked[1], self.W_q12)
        b2 = torch.matmul(b2, self.W_q12_upscale)
        return b1, b2








# class ExtendedMultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, **kwargs):
#         super(ExtendedMultiheadAttention, self).__init__()
#         self.d_model = embed_dim
#         self.nhead = num_heads
#         self.rank = 64
        
#         self.mha = nn.MultiheadAttention(embed_dim, num_heads, **kwargs)
        
#         # Parameters for dw_proj
#         self.W_q1 = nn.Parameter(torch.randn(embed_dim, self.rank * 2))
#         self.W_q2 = nn.Parameter(torch.randn(self.rank * 2, embed_dim))
        
#         # Parameters for compose
#         self.theta = {
#             'W_q1': nn.Parameter(torch.randn(self.rank, embed_dim)),
#             'W_q2': nn.Parameter(torch.randn(self.rank, embed_dim)),
#             'W_k1': nn.Parameter(torch.randn(self.rank, embed_dim)),
#             'W_k2': nn.Parameter(torch.randn(self.rank, embed_dim)),
#             'W_gg': nn.Parameter(torch.randn(num_heads, self.rank)),
#             'W_kg': nn.Parameter(torch.randn(num_heads, self.rank)),
#             'W_qg': nn.Parameter(torch.randn(num_heads, self.rank))
#         }
        
#         # Parameters for post-composition
#         self.theta_pc = nn.Parameter(torch.randn(embed_dim, embed_dim))
    
#     def dw_proj(self, X):
#         B, T, D_m = X.shape
#         W_q1 = self.W_q1
#         W_q2 = self.W_q2
        
#         X = F.gelu(torch.matmul(X, W_q1))
#         X = torch.matmul(X, W_q2)
#         return X.chunk(2, dim=-1)

#     def compose(self, a, b, c, d):
#         theta = self.theta
        
#         W_q1, W_q2, W_k1, W_k2 = theta['W_q1'], theta['W_q2'], theta['W_k1'], theta['W_k2']
#         W_gg, W_kg, W_qg = theta['W_gg'], theta['W_kg'], theta['W_qg']
        
#         dv1_a, dv2_a = self.dw_proj(a)
#         dv1_b, dv2_b = self.dw_proj(b)
#         dv1_c, dv2_c = self.dw_proj(c)
#         dv1_d, dv2_d = self.dw_proj(d)
        
#         o_gg = torch.einsum('bhtd, hr -> bhrt', dv1_a, W_gg) + torch.einsum('bhtd, hr -> bhrt', dv2_a, W_gg)
#         o_kg = torch.einsum('bhtd, hr -> bhrt', dv1_b, W_kg) + torch.einsum('bhtd, hr -> bhrt', dv2_b, W_kg)
#         o_qg = torch.einsum('bhtd, hr -> bhrt', dv1_c, W_qg) + torch.einsum('bhtd, hr -> bhrt', dv2_c, W_qg)
        
#         o_kg = torch.tanh(o_kg)
        
#         return o_gg + o_kg + o_qg

#     def DCMHA(self, query, key, value, causal):
#         B, T, D = query.shape
#         H = self.nhead
        
#         # query, key, value = [rearrange(x, 'B(TD) -> (B T) D') for x in (query, key, value)]
        
#         attn_output, attn_output_weights = self.mha(query, key, value)
        
#         # g = torch.einsum('bhtd, hr -> bhrt', attn_output, self.theta_pc)
#         g = torch.matmul(attn_output, self.theta_pc)
#         # g = g.chunk(3, dim=-1)
        
        
#         logits = self.compose(attn_output, g[0], g[1], g[2])
        
#         probs = logits.softmax(dim=-1)
        
#         return probs, attn_output_weights

#     def forward(self, query, key, value, causal=False, **kwargs):
#         output, attn_output_weights = self.DCMHA(query, key, value, causal)
#         return output, attn_output_weights

# # Helper function for rearranging dimensions (similar to einops.rearrange)
# def rearrange(x, pattern):
#     return x.view(x.shape[0], -1)

