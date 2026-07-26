## Word2Vec SGNS implementation

This repository contains an implementation of word2vec SGNS model using only numpy arrays, the implementation is
based on the original paper [Distributed Representations of Words and Phrases and their
Compositionality](https://arxiv.org/pdf/1310.4546). Code for embedding training is also provided along with some
evaluation results. 


### Embedding Visualization
The embeddings were trained on [the dataset](https://huggingface.co/datasets/sentence-transformers/reddit-title-body)
containing posts from Reddit. The visualization [here](https://daria-kay.github.io/word2vec/visualization/word2vec_w7_p30.html)
shows a tSNE projection of 100-dimensional word embeddings on a plane.

