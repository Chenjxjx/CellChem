from torch import nn,Tensor
import torch
from typing import Dict, Mapping, Optional, Tuple, Any, Union
import numpy as np
import math
class Generation(nn.Module):
    def __init__(self, model_scGPT,graphtransformer):
        super().__init__()
        self.scgpt = model_scGPT
        self.mol_encoder = graphtransformer
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.gamma_proj = nn.Linear(512, 512)  # FiLM: gamma
        self.beta_proj = nn.Linear(512, 512)   # FiLM: beta
        self.fuse_ln = nn.LayerNorm(512)
        self.pred_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)  
        )


    def forward(self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        data):
        
  
        cell_embedding = self.scgpt._encode(
                src,
                values.float(),
                src_key_padding_mask=src_key_padding_mask,
            )  # [batch, seq_len, 512]
        # GraphTransformer 
        ris, zis = self.mol_encoder(
            data.x, data.batch, 
            data.edge_index, data.edge_attr,
            data.get('chem_desc', None)
        )  # ris: [batch, 512]

        # FiLM 
        gamma = self.gamma_proj(ris).unsqueeze(1).expand(-1, cell_embedding.size(1), -1)
        beta = self.beta_proj(ris).unsqueeze(1).expand(-1, cell_embedding.size(1), -1)
        x = gamma * cell_embedding + beta  # [batch, seq_len, 512]

        #Cross-modal attention: with gene tokens as queries and molecular embeddings as keys/values
        q = self.q_proj(x)  # [batch, seq_len, 512]
        k = self.k_proj(ris).unsqueeze(1)  # [batch, 1, 512]
        v = self.v_proj(ris).unsqueeze(1)  # [batch, 1, 512]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
        attended = attn_weights * v.expand(-1, x.size(1), -1)  # [batch, seq_len, 512]
        fused = self.fuse_ln(x + attended)  # [batch, seq_len, 512]
        pred = self.pred_head(fused).squeeze(-1)  # [batch, seq_len]

        return cell_embedding, ris, pred

