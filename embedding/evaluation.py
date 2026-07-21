import numpy as np
import polars as pl

from embedding import Word2Vec

def _calc_mean_cosine_dist(a_matrix, b_matrix):
    return (np.sum(a_matrix * b_matrix, axis=1) / (np.linalg.norm(a_matrix, axis=1) * np.linalg.norm(b_matrix, axis=1))).mean()

def _eval_embeddings(word2vec: Word2Vec, benchmark_filename: str):
    word_cols = ['word1', 'word2', 'word3', 'target']
    words_in_vocab = pl.all_horizontal(
        pl.col(word_cols).map_elements(lambda word: word in word2vec, return_dtype=pl.Boolean)
    )
    known_word_pairs = (
        pl.read_csv(benchmark_filename)
        .filter(words_in_vocab)
        .select(pl.col(['word1', 'word2', 'word3', 'target']).map_elements(lambda word: word2vec[word], return_dtype=pl.List(pl.Float64)))
    )
    known_word_pairs = np.stack([np.stack(known_word_pairs[col].to_list()) for col in word_cols], axis=1)
    diff1 = known_word_pairs[:, 0, :] - known_word_pairs[:, 1, :]  # n x emb_dim
    diff2 = known_word_pairs[:, 2, :] - known_word_pairs[:, 3, :]  # n x emb_dim
    return _calc_mean_cosine_dist(diff1, diff2)

def eval_syntactic_relations(word2vec: Word2Vec):
    return _eval_embeddings(word2vec, 'data/msr.csv')

def eval_semantic_relations(word2vec: Word2Vec):
    return _eval_embeddings(word2vec, 'data/semeval.csv')
