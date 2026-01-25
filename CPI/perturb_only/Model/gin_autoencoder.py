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
        self.MaxPool1d = nn.MaxPool1d(978)
        self.MaxPool1d_seq =  nn.MaxPool1d(1024)
        encoder_config = BertConfig.from_json_file('./Model/bert.json')
        self.CMAPEncode = BertModel(config=encoder_config, add_pooling_layer=False)
        encoder_config_smiles = BertConfig.from_json_file('./Model/bert_smiles.json')
        self.CRISPREncode = BertModel(config=encoder_config_smiles, add_pooling_layer=False)
        self.h_linear = nn.Linear(self.emb_dim, self.feat_dim)   #128->64
        self.h_linear_cmap = nn.Linear(self.emb_dim, self.feat_dim)   #128->64
        self.cross= CrossAttention(self.hid_dim, self.n_heads, self.dropout)
        self.csm_head = nn.Linear(128,32) 
        self.fc = nn.Linear(32,2)
        self.relu =F.relu
        self.loss = nn.BCELoss()
        self.sig = nn.Sigmoid()
        
    def forward(self, CRISPR,Cmap,label):
        cmap_embeds = self.CMAPEncode(Cmap,return_dict = True, mode = 'text')           
        crispr_embeds = self.CRISPREncode(CRISPR,return_dict = True, mode = 'text')   
        cmap_embeds_mean = cmap_embeds.last_hidden_state[:,0,:]
        crispr_embeds_mean = crispr_embeds.last_hidden_state[:,0,:]
        vl_embeddings = self.cross(cmap_embeds_mean, crispr_embeds_mean)
        vl_output_1 = self.relu(self.csm_head(vl_embeddings))
        vl_output = self.fc(vl_output_1)
        vl = self.sig(vl_output)
        one_hot_labels =torch.nn.functional.one_hot(label, num_classes=2)
        Loss = self.loss(vl,one_hot_labels.float())
        vl_1 = vl.detach().cpu().numpy()
        ys = F.softmax(vl, 1).to('cpu').data.numpy()
        predicted_labels = np.argmax(ys, axis=1)
        predicted_scores = ys[:, 1]


        
        
        
        
        return   Loss, predicted_labels,predicted_scores
 