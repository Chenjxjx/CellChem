import torch
from torch import nn
import torch.nn.functional as F
import math
import random
from Model.bert import BertConfig, BertModel, BertLMHeadModel
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import sklearn.metrics
import sklearn
from transformers import BertTokenizer
import regex as re
import numpy as np
import esm
from torch import autograd
    
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x        

class Attention(nn.Module):
    """ A class for attention mechanisn with QKV attention """
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        
        
        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.linear = nn.Linear(hid_dim, hid_dim)

        self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout))
            
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to('cuda')

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

        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)


        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        a=F.softmax(energy, dim=-1)
        
        attention = self.drop(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, -1, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.out(weighter_matrix)
        

        return weighter_matrix
    
    
    
class CrossAttention(nn.Module):
    """
        The main idea of Perceiver CPI (cross attention block + self attention block).
    """

    def __init__(self, hid_dim, n_heads, dropout):

        super(CrossAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.att = Attention(self.hid_dim, self.n_heads,self.dropout)

    
    def forward(self,x,y):     
        output = self.att(x,y,y)

        return output
        
    
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

        
    

class SMILES_CMAP_CL(nn.Module):

    def __init__(self, num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, gene_number=978,embedding_dim = 32,bidirectional=False, dim=256,num_layers=5,num_gc_layers=5, AUTO_activation='mish', queue_size = 1024, hid_dim=64, n_heads=2, dropout=0.1,dim_head=16):
        super(SMILES_CMAP_CL, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.gene_number = gene_number
        self.bidirectional = bidirectional
        self.dim = dim
        self.queue_size = queue_size
        self.num_layers = num_layers
        self.AUTO_activation=AUTO_activation
        self.batchnorm=True
        self.hid_dim = hid_dim
        self.dim_head = dim_head
        self.n_heads = n_heads
        self.dropout = dropout
        self.MaxPool1d_seq =  nn.MaxPool1d(1024)
        self.cross_seq = CrossAttention(1280, self.n_heads, self.dropout)
        self.head_seq = nn.Linear(873,32)
        self.protein_linear = nn.Linear(1280,300) 
        self.mol_linear = nn.Linear(300,1280)
        self.fc_seq = nn.Linear(32,2)
        self.relu =F.relu
        self.loss = nn.BCELoss()
        self.sig = nn.Sigmoid()
        #self.protein_linear_dim1 = nn.Linear(1024,512) 
        #self.mol_linear_dim1 = nn.Linear(873,512) 
        
    def forward(self,Protein,Mol,label):
        #print(Mol.shape) #(128,873,300)
        #Protein = Protein.permute(0, 2, 1)
        #Protein = self.protein_linear_dim1(Protein)
        #Protein = Protein.permute(0, 2, 1)
        #print(Protein.shape) #(b,1024,1280)
        #Protein_him = self.protein_linear(Protein)  ###(b,1024,300)
        Mol_him = self.mol_linear(Mol)
        #Mol = Mol.permute(0, 2, 1)
        #Mol = self.mol_linear_dim1(Mol)
        #Mol = Mol.permute(0, 2, 1)    ###(b,512,300)

        # compound_mask, protein_mask = self.make_masks(Mol_len, Protein_len, compound_max_len, protein_max_len)
        
        #print(Protein_him.shape)
        seq_embeddings = self.cross_seq(Mol_him,Protein)
        #seq_embeddings = seq_embeddings.permute(0, 2, 1)
        seq_output_1 = torch.norm(seq_embeddings, dim=2)
        seq_output_1 = F.softmax(seq_output_1, dim=1)
        #seq_output_1 = self.MaxPool1d_seq(seq_embeddings).squeeze(2)
        #print(norm.shape)
        seq_output_1 = self.relu(self.head_seq(seq_output_1))
        seq_output = self.fc_seq(seq_output_1)
        seq = self.sig(seq_output)
        one_hot_labels =torch.nn.functional.one_hot(label, num_classes=2)
        Loss = self.loss(seq,one_hot_labels.float())
        seq_1 = seq.detach().cpu().numpy()
        seq_1 = np.argmax(seq_1,axis=1)
        
        return   Loss, seq_1