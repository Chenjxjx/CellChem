import sys
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from .gt_layer import GraphTransformerLayer

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

class DescAttention(nn.Module):
    def __init__(
        self, 
        num_descriptors = -1, 
        hidden_dim = None,
        emb_dim = 512,
        num_heads = 8,
        dropout = 1,
    ):
        super(DescAttention, self).__init__()
        self.num_descriptors = num_descriptors
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        if hidden_dim is None:
            hidden_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.q = nn.Linear(num_descriptors, hidden_dim * num_heads)
        self.k = nn.Linear(num_descriptors, hidden_dim * num_heads)
        self.v = nn.Linear(num_descriptors, hidden_dim * num_heads)
        
        self.inv_sqrt = 1 / (hidden_dim ** 0.5)
        self.dropout = dropout
        
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, emb_dim), 
            nn.ReLU(inplace = True),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),    
        )
        
    def forward(
        self, 
        x
    ):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.view(-1, self.num_heads, 1, self.hidden_dim)
        k = k.view(-1, self.num_heads, self.hidden_dim, 1)
        v = v.view(-1, self.num_heads, self.hidden_dim)
        score = torch.matmul(q, k).squeeze()
        
        if self.dropout < 1:
            score = F.dropout(x, self.dropout, training = self.training)
        
        x = v * score.unsqueeze(-1)
        
        x = x.view(-1, self.num_heads * self.hidden_dim)
        
        return self.lin(x)

class GraphTransformerConv(MessagePassing):
    def __init__(
        self, 
        emb_dim: int = 320,
        num_heads: int = 8,
        drop_ratio: float = 0, 
        hidden_mult: float = 1.,
        act: str = 'silu',
        **kwargs,
    ):
        super(GraphTransformerConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.GTlayer = GraphTransformerLayer(
            emb_dim,
            emb_dim,
            num_heads,
            drop_ratio,
            hidden_mult,
            act = act,
            **kwargs,
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, batch, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        
        h, e = self.GTlayer(x, batch, edge_index, edge_embeddings)
        return self.propagate(edge_index, x = h, edge_attr = e)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class DescAttention(nn.Module):
    def __init__(
        self, 
        num_descriptors = -1, 
        hidden_dim = None,
        emb_dim = 512,
        num_heads = 8,
        dropout = 1,
    ):
        super(DescAttention, self).__init__()
        self.num_descriptors = num_descriptors
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        if hidden_dim is None:
            hidden_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.q = nn.Linear(num_descriptors, hidden_dim * num_heads)
        self.k = nn.Linear(num_descriptors, hidden_dim * num_heads)
        self.v = nn.Linear(num_descriptors, hidden_dim * num_heads)
        
        self.inv_sqrt = 1 / (hidden_dim ** 0.5)
        self.dropout = dropout
        
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, emb_dim), 
            nn.ReLU(inplace = True),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),    
        )
        
    def forward(
        self, 
        x
    ):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.view(-1, self.num_heads, 1, self.hidden_dim)
        k = k.view(-1, self.num_heads, self.hidden_dim, 1)
        v = v.view(-1, self.num_heads, self.hidden_dim)
        score = torch.matmul(q, k).squeeze()
        
        if self.dropout < 1:
            score = F.dropout(x, self.dropout, training = self.training)
        
        x = v * score.unsqueeze(-1)
        
        x = x.view(-1, self.num_heads * self.hidden_dim)
        
        return self.lin(x)

class GraphTransformer(nn.Module):
    """
    Args:
        num_layer (int): the number of GraphTransformerLayer layers
        emb_dim (int): the dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
    Output:
        node representations
    """
    def __init__(
        self, 
        num_layer: int = 5, 
        num_descriptors: int = -1,
        num_heads: int = 8,
        emb_dim: int = 300, 
        feat_dim: int = 256, 
        hidden_mult: float = 2.,
        drop_ratio: float = 0, 
        act: str = 'silu',
        pool: str = 'mean',
        desc_att: bool = True,
        **kwargs,
    ):
        super(GraphTransformer, self).__init__()
        self.num_descriptors = num_descriptors
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        
        # add the chemical descriptors
        if num_descriptors > 0:
            if desc_att:
                self.desc_lin = DescAttention(
                    num_descriptors, 
                    None,
                    emb_dim,
                    num_heads,
                    drop_ratio,
                )
            else:
                self.desc_lin = nn.Sequential(
                    nn.Linear(num_descriptors, emb_dim), 
                    nn.ReLU(inplace = True),
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim),  
                    nn.Linear(emb_dim, emb_dim), 
                    nn.ReLU(inplace = True),
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim),    
                )
        
        # List of MLPs
        self.gtlayers = nn.ModuleList()
        for layer in range(num_layer):
            self.gtlayers.append(GraphTransformerConv(
                emb_dim,
                num_heads,
                drop_ratio,
                hidden_mult,
                act = act,
                **kwargs,
            ))
        
        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )

    def forward(self, x, batch, edge_index, edge_attr, chem_desc = None):
        #print(f"x shape: {x.shape}")
        batch = batch.long()
        #print(f"batch shape: {batch.shape}")
        edge_index = edge_index.long()
        #print(f"edge_index shape: {edge_index.shape}")
        # print(f"edge_attr shape: {edge_attr.shape}")
        
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        
        if self.num_descriptors > 0:
            chem_desc = self.desc_lin(chem_desc)
            h += chem_desc[batch]
        
        for layer in range(self.num_layer):
            h = self.gtlayers[layer](h, batch, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(
                    h, 
                    self.drop_ratio, 
                    training = self.training,
                )
            else:
                h = F.dropout(
                    F.relu(h), 
                    self.drop_ratio, 
                    training = self.training,
                )

        h = self.pool(h, batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        
        return h, out
