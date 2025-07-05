import requests
from pathlib import Path

#Making directory for the data
dir=Path("files")
dir.mkdir(parents=True, exist_ok=True)

#Getting Data from the URL
req=requests.get("https://raw.githubusercontent.com/neychev/"
     "small_DL_repo/master/datasets/Multi30k/training.tar.gz")
with open("files/training.tar.gz", "wb") as f:
    f.write(req.content)


import tarfile
#To read TarFile->And extract the TAR file
train=tarfile.open('files/training.tar.gz')    #C
train.extractall('files',filter="fully_trusted")    #D
train.close()



#Train de is the -> German language Phrases
#Train en is the ->English Language Phrases
with open("./files/train.de", "rb") as f:
    trainde=f.readlines()

with open("./files/train.en", "rb") as f:
    trainen=f.readlines()

#Convertin the above read files to a list of Phrases
trainde=[i.decode("utf-8").strip() for i in trainde]
trainen=[i.decode("utf-8").strip() for i in trainen]



#Tokenization Part of the data, where each phrase in this list will be Broken down into list of token
#Tokens-> The words in the Sequence

import spacy
#Below are the english and German Language Spacy libraries for Creating Tokens

#If there is an Error im loading the Tokens
#In the terminal Run -> pythom -m spacy download de_core_news_sm en_core_web_sm
de_tokenizer = spacy.load("de_core_news_sm")
en_tokenizer = spacy.load("en_core_web_sm")


#Now we finally Tokenize eachh line in our list of data

#BOS-> beginning of Sequence
#EOS-> Ending of Sequence
token_full_en=[["BOS"]+[tok.text for tok in en_tokenizer.tokenizer(x)]+["EOS"] for x in trainen ]
token_full_de=[["BOS"]+[tok.text for tok in de_tokenizer.tokenizer(x)]+["EOS"] for x in trainde ]



