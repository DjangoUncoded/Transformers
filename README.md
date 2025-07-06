
# End-to-End Transformer Architecture for German-to-English Translation


A sleek, modular implementation of a Transformer-based German-to-English translator, inspired by the 2017 landmark paper Attention Is All You Need. This model harnesses the revolutionary self‑attention mechanism that freed sequence models from the constraints of recurrence and convolution, enabling parallel processing, better handling of long-range dependencies, and dramatic improvements in translation quality and training efficiency 


## Introduction

 “Attention Is All You Need” — Core Insights
Transformer Architecture
Introduces a novel neural network that relies solely on attention mechanisms, eliminating recurrence and convolution—ushering in dramatically faster parallel processing and simplified structure.

Self-Attention Mechanism
Each token in an input sequence attends to every other via scaled dot‑product attention—computing Query, Key, and Value vectors (Q, K, V), then weighting values based on softmax‑scaled dot-products.

Multi‑Head Attention
Multiple parallel attention “heads” allow the model to capture diverse relationships—improving its ability to recognize syntax, semantics, and long-range dependencies.

Positional Encoding
Adds either sinusoidal or learned embeddings to token vectors to preserve word order—critical since attention alone is permutation-invariant 



## About the Repository
This repository contains a clean and modular implementation of a Transformer model for language translation. The project breaks down the entire architecture into well-structured Python modules for easy understanding and experimentation:

📦 Data_Tokenization.py: Handles dataset loading and tokenization.

🧾 Dictionaries.py: Manages vocabulary-to-index mappings.

🔁 Encoder_Decoder.py: Defines the encoder and decoder blocks.

🧱 Masking_Batching.py: Creates batches and attention masks for training.

📐 Embeddings_PosEncoding.py: Builds input embeddings and applies positional encoding.

🎯 Multi_Head_Attention.py: Implements the attention mechanism.

⚙️ Transformer.py: Wraps the encoder-decoder architecture into a full model.

🧠 Training.py: Trains the Transformer model.

🌍 Translator.py: Performs translation using the trained model.

📓 NoteBook_WorkFlow.ipynb: Demonstrates how all modules fit together for end-to-end training and inference.


## Acknowledgements
This project is heavily inspired by the Transformer model explained and implemented in the following repository:


 - [Link to Original GitHub Repository ](https://github.com/markhliu/txt2img/blob/main/ch03_NEW_VisionTransformer.ipynb)


