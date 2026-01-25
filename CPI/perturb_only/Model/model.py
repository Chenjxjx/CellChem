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


    
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x        

class Attention(nn.Module):
    """ A class for attention mechanisn with QKV attention """
    def __init__(self, hid_dim, dim_head, n_heads, dropout):
        super().__init__()
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.dim_head = dim_head
        self.n_heads = n_heads
        self.inner_dim = self.dim_head * self.n_heads
        
        
        self.f_q = nn.Linear(hid_dim, self.inner_dim)
        self.f_k = nn.Linear(hid_dim, self.inner_dim)
        self.f_v = nn.Linear(hid_dim, self.inner_dim)

        self.linear = nn.Linear(hid_dim, self.inner_dim)

        self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(self.inner_dim, hid_dim),
            nn.Dropout(dropout))
            
        self.scale = dim_head ** -0.5

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

        Q = Q.view(batch_size, self.n_heads,self.dim_head).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.dim_head).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.n_heads, self.dim_head).unsqueeze(3)

        energy = torch.matmul(Q, K_T) * self.scale
        

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        a=F.softmax(energy, dim=-1)
        
        attention = self.drop(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.inner_dim)

        weighter_matrix = self.out(weighter_matrix)
        

        return weighter_matrix
    
    
    
class CrossAttention(nn.Module):
    """
        The main idea of Perceiver CPI (cross attention block + self attention block).
    """

    def __init__(self, hid_dim, dim_head, n_heads, dropout):

        super(CrossAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.dim_head = dim_head
        self.att = Attention(self.hid_dim, self.dim_head, self.n_heads,self.dropout)

    
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
        
        
        self.CMAPEncode =  nn.Linear(978,512)
        
        self.CRISPREncode = nn.Linear(978,512)
        self.h_linear = nn.Linear(512,64)   #128->64
        self.h_linear_cmap = nn.Linear(512,64)   #128->64
        self.cross= CrossAttention(self.hid_dim, self.dim_head, self.n_heads, self.dropout)
        self.csm_head = nn.Linear(128, 32) 
        self.fc = nn.Linear(32,1)
        self.relu =F.relu
        self.loss = nn.BCEWithLogitsLoss()
        self.o = nn.Sigmoid()
        
    def forward(self, CRISPR,Cmap,label):
        CRISPR=CRISPR.to(torch.float32)
        Cmap = Cmap.to(torch.float32)
        cmap_embeds = self.CMAPEncode(Cmap)           
        cmap_feat =self.relu(cmap_embeds)
        cmap_feat = self.h_linear_cmap(cmap_feat)
        
        crispr_embeds = self.CRISPREncode(CRISPR)   
        #  print(smiles_embeds.last_hidden_state[:,0,:].shape)   ---> torch.Size([64, 128])
        crispr_feat = self.relu(crispr_embeds)
        crispr_feat = self.h_linear(crispr_feat)
        #print(smiles_feat.shape)    ---> torch.Size([64, 64])
        vl_embeddings = torch.cat([cmap_feat,crispr_feat],dim=1)
        #vl_embeddings = self.cross(cmap_feat, crispr_feat)
        vl_output_1 = self.relu(self.csm_head(vl_embeddings))
        vl_output = self.fc(vl_output_1)
        vl_output = vl_output.squeeze(dim=-1)  
        #ne_hot_labels = torch.nn.functional.one_hot(label, num_classes=2).long()
        loss = self.loss(vl_output,label.float())
        #loss = F.cross_entropy(vl_output,label) 
        vl = self.o(vl_output).detach().cpu().numpy()
        
        up_feat=torch.where(CRISPR_up>0, 1., CRISPR_up)
        label_1 = label.detach().cpu().numpy()
        #print(CRISPR,Cmap)
        #print(v1,label_1)
        #ACC = 0
        print(vl)
        #ACC = sklearn.metrics.accuracy_score(label_1,v1)
        return loss,ACC