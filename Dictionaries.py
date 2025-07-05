#Here the objectives is to create two dictionaries
#One which will map a word to its index->useful for Training and Encoding
#One which will map the indexes to the word, used to Generate the actual words

from Data_Tokenization import token_full_en,token_full_de
from collections import Counter

#English Dictionary

en_word_count=Counter()
for sent in token_full_en:
    for word in sent:
        en_word_count[word]+=1
PAD=0
UNK=1
frequency_en=en_word_count.most_common(50000)
en_total_word=len(frequency_en)+2
#Counter-> Returns the word Itself and the count

en_word_dict={w[0]:idx+2 for idx,w in enumerate(frequency_en)}
en_word_dict["PAD"]=PAD
en_word_dict["UNK"]=UNK
en_idx_dict={v:k for k,v in en_word_dict.items()}

#German Dictionary

de_word_count=Counter()
for sent in token_full_de:
    for word in sent:
        de_word_count[word]+=1
de_frequency=de_word_count.most_common(50000)
de_total_word=len(de_frequency)+2
de_word_dict={w[0]:idx+2 for idx,w in enumerate(de_frequency)}
de_word_dict["PAD"]=PAD
de_word_dict["UNK"]=UNK
de_idx_dict={v:k for k,v in de_word_dict.items()}



#Fina dictionary Which converts All the tokens(word) to the respective Index

out_en_ids=[[en_word_dict.get(w,UNK) for w in s]
            for s in token_full_en]
out_de_ids=[[de_word_dict.get(w,UNK) for w in s]
            for s in token_full_de]


#We will sort the above list of sequences (in form of indexes as words) in the ascending order
#of the length german sequences
sorted_ids=sorted(range(len(out_de_ids)),
                  key=lambda x:len(out_de_ids[x]))

out_de_ids=[out_de_ids[x] for x in sorted_ids]
out_en_ids=[out_en_ids[x] for x in sorted_ids]