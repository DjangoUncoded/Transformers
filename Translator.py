from Training import *


#The input to the model is a simple German phrase
#The model will convert it to the equivalent English Phrase

from Masking_Batching import subsequent_mask
from Data_Tokenization import de_tokenizer
def de2en(ger):
    tokenized_ger= [tok.text for tok in de_tokenizer.tokenizer(ger)]
    tokenized_ger=["BOS"]+tokenized_ger+["EOS"]
    geridx=[de_word_dict.get(i,UNK) for i in tokenized_ger]
    src=torch.tensor(geridx).long().to(device).unsqueeze(0)
    src_mask=(src!=0).unsqueeze(-2)
    memory=model.encode(src,src_mask)    #A
    start_symbol=en_word_dict["BOS"]
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    translation=[]
    for i in range(100):
        out = model.decode(memory,src_mask,ys,
        subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])    #B
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(
            src.data).fill_(next_word)], dim=1)
        sym = en_idx_dict[ys[0, -1].item()]
        if sym != 'EOS':    #C
            translation.append(sym)
        else:
            break
    trans=" ".join(translation)
    for x in '''?:;.,'("-!&)%''':
        trans=trans.replace(f" {x}",f"{x}")    #D
    return trans



