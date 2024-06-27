import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)
        if self.q_linear.bias is not None:
            nn.init.constant_(self.q_linear.bias, 0)
        if self.k_linear.bias is not None:
            nn.init.constant_(self.k_linear.bias, 0)
        if self.v_linear.bias is not None:
            nn.init.constant_(self.v_linear.bias, 0)
        if self.out_linear.bias is not None:
            nn.init.constant_(self.out_linear.bias, 0)

    def scaled_dot_product_attention(self, Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        batch_size, seq_length, embed_dim = query.size()

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape to (batch_size, num_heads, seq_length, head_dim)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores and scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V)

        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.out_linear(attn_output)

        return output, attn_weights