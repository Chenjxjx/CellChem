import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_softmax
from .apis import get_activation


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            num_heads: int,
            use_bias: bool = False,
            eps: float = 1e-9,
    ):
        super(MultiHeadAttentionLayer, self).__init__()
        assert out_dim % num_heads == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.eps = eps

        self.q = nn.Linear(in_dim, out_dim, bias = use_bias)
        self.k = nn.Linear(in_dim, out_dim, bias = use_bias)
        self.v = nn.Linear(in_dim, out_dim, bias = use_bias)
        self.edge_proj = nn.Linear(in_dim, out_dim, bias = use_bias)

        self.inv_sqrt = 1 / ((out_dim // num_heads) ** 0.5)

    def forward(
            self,
            x: Tensor, # shape (num_nodes, in_dim)
            batch: Tensor, # shape (num_nodes)
            edge_index: Tensor, # shape (2, num_edges)
            edge_feature: Tensor, # shape (num_edges, in_dim)
    ):
        num_nodes = x.size(0)
        # source_to_target flow
        edge_j, edge_i = edge_index
        edge_batch = batch[edge_i]

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        edge_feature = self.edge_proj(edge_feature)

        dim_per_head = self.out_dim // self.num_heads
        q = q.view(-1, self.num_heads, dim_per_head)
        k = k.view(-1, self.num_heads, dim_per_head)
        v = v.view(-1, self.num_heads, dim_per_head)
        edge_feature = edge_feature.view(-1, self.num_heads, dim_per_head)

        # propagate_attention
        # 1. Compute attention score
        k_src = k[edge_j]
        q_dst = q[edge_i]
        score = k_src * q_dst
        # 2. score scaling
        score = score * self.inv_sqrt
        # 3. Use available edge features to modify the scores
        score = score * edge_feature
        # 4. Copy edge features as e_out to be passed to FFN_e
        edge_feature = score
        # 5. softmax
        score = scatter_softmax(score, edge_batch, dim = 0)
        v_src = v[edge_j] * score
        weighted_v = scatter(v_src, edge_i, dim = 0,
                dim_size = num_nodes, reduce = 'sum')
        sum_score = scatter(score, edge_i, dim = 0,
                dim_size = num_nodes, reduce = 'sum')

        x = weighted_v / torch.clamp_min(sum_score, self.eps)
        x = x.view(-1, self.out_dim)
        edge_feature = edge_feature.view(-1, self.out_dim)

        return x, edge_feature # shape (num_nodes, out_dim) (num_edges, out_dim)

class GraphTransformerLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            num_heads: int,
            dropout_rate: float = 0.1,
            hidden_mult: float = 1.,
            layer_norm: bool = True,
            residual: bool = True,
            act: str = 'leakyrelu',
            use_bias: bool = False,
            **kwargs
    ):
        super(GraphTransformerLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.hidden_mult = hidden_mult
        self.layer_norm = layer_norm
        self.residual = residual
        self.use_bias = use_bias

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim,
                                                 num_heads, use_bias)
        self.lin_x = nn.Linear(out_dim, out_dim)
        self.lin_e = nn.Linear(out_dim, out_dim)

        self.norm1_x = nn.LayerNorm(out_dim) if layer_norm else nn.BatchNorm1d(out_dim)
        self.norm1_e = nn.LayerNorm(out_dim) if layer_norm else nn.BatchNorm1d(out_dim)

        # Feed forward for node embedding
        ff_hidden_dim = int(out_dim * hidden_mult)
        self.ff_x_1 = nn.Linear(out_dim, ff_hidden_dim)
        self.ff_x_2 = nn.Linear(ff_hidden_dim, out_dim)
        self.act_x = get_activation(act)(**kwargs)
        # Feed forward for edge embedding
        self.ff_e_1 = nn.Linear(out_dim, ff_hidden_dim)
        self.ff_e_2 = nn.Linear(ff_hidden_dim, out_dim)
        self.act_e = get_activation(act)(**kwargs)
        self.norm2_x = nn.LayerNorm(out_dim) if layer_norm else nn.BatchNorm1d(out_dim)
        self.norm2_e = nn.LayerNorm(out_dim) if layer_norm else nn.BatchNorm1d(out_dim)

    def forward(
            self,
            x: Tensor,  # shape (num_nodes, in_dim)
            batch: Tensor,  # shape (num_nodes)
            edge_index: Tensor,  # shape (2, num_edges)
            edge_feature: Tensor,  # shape (num_edges, in_dim)
    ):
        # Copy for 1st residual connection
        _x = x
        _e = edge_feature
        # print(f"GT: x shape: {x.shape}")
        # print(f"GT: batch shape: {batch.shape}")
        # print(f"GT: edge_index shape: {edge_index.shape}")
        # print(f"GT: edge_feature shape: {edge_feature.shape}")
        x, e = self.attention(x, batch, edge_index, edge_feature)
        x = F.dropout(x, self.dropout_rate, training = self.training)
        e = F.dropout(e, self.dropout_rate, training = self.training)
        x = self.lin_x(x)
        e = self.lin_e(e)

        if self.residual:
            x = x + _x
            e = e + _e
        x = self.norm1_x(x)
        e = self.norm1_e(e)

        # Copy for 2nd residual connection
        _x = x
        _e = e
        # ff for node embedding
        x = self.ff_x_1(x)
        x = self.act_x(x)
        x = F.dropout(x, self.dropout_rate, training = self.training)
        x = self.ff_x_2(x)
        # ff for edge embedding
        e = self.ff_e_1(e)
        e = self.act_e(e)
        e = F.dropout(e, self.dropout_rate, training=self.training)
        e = self.ff_e_2(e)

        if self.residual:
            x = x + _x
            e = e + _e
        x = self.norm2_x(x)
        e = self.norm2_e(e)

        return x, e
