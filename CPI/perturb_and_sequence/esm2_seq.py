import esm
import torch
import os
from typing import List, Tuple
import string
import torch.nn.functional
import numpy as np
import random
from torch import nn
import pandas as pd

Result = pd.read_csv('./data/dataset.csv')
data_list = Result['seq']
def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

esm1b, esm1b_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm1b = esm1b.eval().cuda()
esm1b_batch_converter = esm1b_alphabet.get_batch_converter()
list1=[]

a=0
for i in range(len(data_list)):
    a+=1
    esm1b_data = [(data_list[i],data_list[i])]
    esm1b_batch_labels, esm1b_batch_strs, esm1b_batch_tokens = esm1b_batch_converter(esm1b_data)
    b =esm1b_batch_tokens.shape[1:3][0]
    if b>1024:
        tokens=esm1b_batch_tokens[:,0:1024]
    else:
        tokens=esm1b_batch_tokens
    #elif b<=1024:
    #    tokens=torch.nn.functional.pad(esm1b_batch_tokens,pad=(0,1024-b,0,0), mode='constant', value=1)
    tokens = torch.tensor(tokens.numpy())
    with torch.no_grad():
        tokens=tokens.cuda()
        results = esm1b(tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33].cpu()
        np.save('./protein_embedding_esm2_max1024/'+Result['protein'][i]+'.npy',token_representations.numpy())
    torch.cuda.empty_cache()
    print(Result['protein'][i],token_representations.shape)
    torch.cuda.empty_cache()


