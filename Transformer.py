from  torch import nn
#Here the encoder ,decoder, MultiHeadAttention ,Generator Classes will be initiated

from Multi_Head_Attention import *
from Embeddings_PosEncoding import *
from Dictionaries import *
from Encoder_Decoder import *
device="cuda"

class Transformer(nn.Module):
    def __init__(self, encoder, decoder,
                 src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt),
                            memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        output = self.decode(memory, src_mask, tgt, tgt_mask)
        return output



def create_model(src_vocab, tgt_vocab, N, d_model,
                 d_ff, h, dropout=0.1):
    attn=MultiHeadedAttention(h, d_model).to(device)

    ff=PositionwiseFeedForward(d_model, d_ff, dropout).to(device)

    pos=PositionalEncoding(d_model, dropout).to(device )

    model = Transformer(
        Encoder(EncoderLayer(d_model,deepcopy(attn),deepcopy(ff),
                             dropout).to(device),N).to(device),
        Decoder(DecoderLayer(d_model,deepcopy(attn),
             deepcopy(attn),deepcopy(ff), dropout).to(device),
                N).to(device),
        nn.Sequential(Embeddings(d_model, src_vocab).to(device),
                      deepcopy(pos)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(device),
                      deepcopy(pos)),
        Generator(d_model, tgt_vocab)).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(device)


src_vocab=len(de_word_dict)
trg_vocab=len(en_word_dict)


model = create_model(src_vocab, trg_vocab, N=6,
    d_model=256, d_ff=1024, h=8, dropout=0.1)

#