import torch

from Dictionaries import out_en_ids,out_de_ids
device="cuda"
from Data_Tokenization import token_full_de

PAD=0
import numpy as np
#Dividing the whole data into batches of size 126 each

batch_size=128
idx_list=np.arange(0,len(token_full_de),batch_size)
np.random.shuffle(idx_list)


#Below List contains the list of list, where the inner list is of size 128, where there are indexes
#Of the Sequences (Phrases)
batch_indexs=[]
for idx in idx_list:
    batch_indexs.append(np.arange(idx,min(len(token_full_de),idx+batch_size)))


#Supposedly in a batch if sequences are of different length, the padding takes care of
#This irregularity, by appending zeros to the other sequences to match the length of the longest Sequence
def seq_padding(X, padding=PAD):
    L = [len(x) for x in X]
    ML = max(L)
    padded_seq = np.array([np.concatenate([x,
                   [padding] * (ML - len(x))])
        if len(x) < ML else x for x in X])
    return padded_seq


#This makes sure that the transformer is auto-regressive during training and inference
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape),
                              k=1).astype('uint8')
    output = torch.from_numpy(subsequent_mask) == 0
    return output

def make_std_mask(tgt, pad):
    tgt_mask=(tgt != pad).unsqueeze(-2)
    output=tgt_mask & subsequent_mask(\
        tgt.size(-1)).type_as(tgt_mask.data)
    return output

class Batch:
    def __init__(self, src, trg=None, pad=0):
        src = torch.from_numpy(src).to(device).long()
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            trg = torch.from_numpy(trg).to(device).long()
            #Targ is the decoder input during training
            self.trg = trg[:, :-1]

            #Targ_y is the decoder expected output->during testing
            #Non Auto Regressive
            self.trg_y = trg[:, 1:]

            self.trg_mask = make_std_mask(self.trg, pad)
            #ntokens counts the number of real tokens
            self.ntokens = (self.trg_y != pad).data.sum()


#In the Final Training we will be passing the objects of the Above Class batch
batches=[]
for b in batch_indexs:

    batch_en=[out_en_ids[x] for x in b]
    batch_de=[out_de_ids[x] for x in b]

    batch_en=seq_padding(batch_en)
    batch_de=seq_padding(batch_de)
    batches.append(Batch(batch_de,batch_en))