from torch import nn
import torch
import math

class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super().__init__()
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model

    def forward(self, x):
        out=self.lut(x)*math.sqrt(self.d_model)
        return out


#%%
#INPUT to Positional Encoding will be the output of Embeddings of the sentences

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0., max_len,
                                device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0., d_model, 2, device=device)
            * -(math.log(10000.0) / d_model))
        pe_pos = torch.mul(position, div_term)
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        out = self.dropout(x)
        return out