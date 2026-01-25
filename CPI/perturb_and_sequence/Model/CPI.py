import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .gt_layer import GraphTransformerLayer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
# from Radam import *
# from lookahead import Lookahead
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

def load_tensor(file_name, dtype):
    return [dtype(d).to('cuda:0') for d in np.load(file_name + '.npy',allow_pickle=True)]

class SelfAttention(nn.Module):
    """ A class for attention mechanisn with QKV attention """
    def __init__(self, hid_dim, n_heads, dropout,device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value, mask=None):
        """ 
        :Query : A projection function
        :Key : A projection function
        :Value : A projection function
        Cross-Att: Key and Value should always come from the same source (Aiming to forcus on), Query comes from the other source
        Self-Att : Both three Query, Key, Value come from the same source (For refining purpose)
        """
        batch_size = query.shape[0]
        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)
        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        energy = torch.matmul(Q, K_T) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        weighter_matrix = torch.matmul(attention, V)
        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()
        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))
        weighter_matrix = self.do(self.fc(weighter_matrix))
        return weighter_matrix

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
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
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
        batch = batch.long()
        edge_index = edge_index.long()
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
        return out

class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        protein_emb = self.fc(protein)
        protein_emb = protein_emb.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(protein_emb))
            conved = F.glu(conved, dim=1)
            protein_emb = conved
        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        return trg

class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, Mol_encoder, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = DecoderLayer(hid_dim, n_heads, pf_dim, self_attention, dropout, device)
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(640, 256)
        self.fc_2 = nn.Linear(256, 2)
        self.gn = nn.GroupNorm(8, 256)
        self.Protein_max_pool = nn.MaxPool1d(873)
        self.mol_encoder = Mol_encoder
        
    def forward(self, trg, src, trg_mask=None,src_mask=None):
        trg =  self.mol_encoder(trg.x, trg.batch, 
            trg.edge_index, trg.edge_attr,
            trg.get('chem_desc', None))
        trg = self.ft(trg)
        trg = self.layers(trg, src,trg_mask,src_mask)
        label = F.relu(self.fc_1(trg))
        label = self.fc_2(label)
        return label


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=300):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.Protein_max_pool = nn.AvgPool1d(1024)
        self.Mol_max_pool = nn.MaxPool1d(873)
        self.fc1_xt = nn.Linear(655360,640)
        self.do = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(5,1)
        
    def normalization(self,vector_present,threshold=0.1):
        vector_present_clone = vector_present.clone()
        num = vector_present_clone - vector_present_clone.min(1,keepdim = True)[0]
        de = vector_present_clone.max(1,keepdim = True)[0] - vector_present_clone.min(1,keepdim = True)[0]
        return num / de


    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num)  # batch size
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return compound_mask, protein_mask


    def forward(self, compound, protein,correct_interaction,atom_num,protein_num):
        enc_src = self.encoder(protein)
        enc_src = enc_src.permute(0, 2, 1)
        enc_src = self.Protein_max_pool(enc_src)
        enc_src = enc_src.permute(0, 2, 1).squeeze(1)
        predicted_interaction = self.decoder(compound, enc_src)
        Loss = nn.CrossEntropyLoss()
        loss = Loss(predicted_interaction, correct_interaction)
        ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
        predicted_labels = np.argmax(ys, axis=1)
        predicted_scores = ys[:, 1]
        return predicted_labels,predicted_scores, loss




