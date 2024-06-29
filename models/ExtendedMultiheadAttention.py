import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class ExtendedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(ExtendedMultiheadAttention, self).__init__()
        self.d_model = embed_dim
        self.nhead = num_heads

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, **kwargs)
        
        # Learnable matrix to modify attention output
        self.learnable_matrix = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, query, key, value, **kwargs):
        attn_output, attn_output_weights = self.mha(query, key, value, **kwargs)
        
        # Apply learnable matrix transformation
        output = torch.matmul(attn_output, self.learnable_matrix)
        
        return output, attn_output_weights


