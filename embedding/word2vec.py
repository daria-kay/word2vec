import os
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from tqdm.notebook import tqdm

from nn import BinaryCrossEntropy, SkipGram, SGD, Optimizer
from . import Vocab


def cosine_dist(first, second) -> np.ndarray:
    if len(first.shape) == 1:
        first = first[None, :]
    if len(second.shape) == 1:
        second = second[None, :]
    return 1.0 - (np.sum(first * second, axis=1) / (np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)))


class Word2VecDataloader(Iterable):

    def __init__(self,
                 data: pl.LazyFrame,
                 vocab: Vocab,
                 batch_size: int,
                 center_word_col: str = 'center',
                 context_word_col: str = 'context',
                 label_col: str = 'label',
                 ):
        self._data: pl.LazyFrame = data
        self._vocab: Vocab = vocab
        self.batch_size: int = batch_size
        self.center_word_col = center_word_col
        self.context_word_col = context_word_col
        self.label_col = label_col

    def __iter__(self):
        for batch in self._data.collect_batches(chunk_size=self.batch_size, maintain_order=True, lazy=True):
            center_words = batch[self.center_word_col].map_elements(lambda word: self._vocab.get_id(word)).to_numpy()
            context_words = batch[self.context_word_col].map_elements(lambda word: self._vocab.get_id(word)).to_numpy()
            labels = batch[self.label_col].to_numpy()

            yield center_words, context_words, labels


class Word2Vec:
    _VOCAB_FILENAME: str = 'vocab.pkl'
    _EMBEDDINGS_FILENAME: str = 'embeddings.pkl'
    _DATASET_FILENAME: str = 'dataset.csv'

    def __init__(self, embedding_size: int, vocab: Vocab, embeddings: np.ndarray):
        self.embedding_size = embedding_size
        self._vocab = vocab
        self._embeddings = embeddings

    def __getitem__(self, word):
        return self._embeddings[self._vocab.get_id(word)]

    def __contains__(self, word):
        return word in self._vocab

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def embedding_dim(self):
        return self._embeddings.shape[-1]

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def words(self):
        return self._vocab.words

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str) -> 'Word2Vec':
        with open(Path(pretrained_model_path, cls._VOCAB_FILENAME), 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(Path(pretrained_model_path, cls._EMBEDDINGS_FILENAME), 'rb') as embeddings_file:
            embeddings = pickle.load(embeddings_file)
        return cls(embeddings.shape[-1], vocab, embeddings)

    @classmethod
    def train(cls,
              dataset_path: str,
              embedding_size: int,
              batch_size: int = 10_000,
              val_percent: int = 10,
              epoches: int = 1,
              lr: float = 1e-3
              ) -> 'Word2Vec':
        with open(Path(dataset_path, cls._VOCAB_FILENAME), 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
        dataset = pl.scan_csv(
            Path(dataset_path, cls._DATASET_FILENAME),
            has_header=False,
            with_column_names=lambda _: ['center', 'context', 'label']
        ).with_row_index()
        val_data = Word2VecDataloader(
            dataset.filter(pl.first() % val_percent == 0),
            vocab,
            batch_size
        )
        train_data = Word2VecDataloader(
            dataset.filter(pl.first() % val_percent != 0),
            vocab,
            batch_size,
        )
        skip_gram = SkipGram(len(vocab), embedding_size)
        objective = BinaryCrossEntropy()
        optimizer = SGD(skip_gram, lr=lr)
        Word2Vec._train_loop(skip_gram, objective, optimizer, train_data, val_data, epoches)
        word2vec = cls(embedding_size, vocab, Word2Vec._normalize(skip_gram.central_embeddings.weights))

        return word2vec

    @staticmethod
    def _normalize(matrix: np.ndarray):
        if len(matrix.shape) == 1:
            matrix = matrix[None, :]
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _train_loop(skip_gram: SkipGram,
                    objective: BinaryCrossEntropy,
                    optimizer: Optimizer,
                    train_data: Word2VecDataloader,
                    val_data: Word2VecDataloader,
                    epoches: int = 1, ):
        for _ in tqdm(range(epoches), 'Training embeddings'):
            train_loss = 0.0
            train_samples = 0
            for central, context, labels in train_data:
                optimizer.zero_grad()
                logits = skip_gram(central, context)
                loss = objective.forward(logits, labels)
                train_loss += loss.sum()
                train_samples += loss.shape[0]

                skip_gram.backward(central, context, objective.backward(logits, labels))
                optimizer.step()

            val_loss = 0.0
            val_samples = 0
            for central, context, labels in val_data:
                logits = skip_gram(central, context)
                loss = objective.forward(logits, labels)
                val_loss += loss.sum()
                val_samples += loss.shape[0]

            print(f'Train loss: {(train_loss / train_samples):.3f}, val loss: {(val_loss / val_samples):.3f}')

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(save_directory + '/' + self._VOCAB_FILENAME, 'wb') as vocab_file:
            pickle.dump(self._vocab, vocab_file)
        with open(save_directory + '/' + self._EMBEDDINGS_FILENAME, 'wb') as embeddings_file:
            pickle.dump(self._embeddings, embeddings_file)

    def get_word_id(self, word: str):
        word = word.strip().lower()
        id = self._vocab.get_id(word)
        if id == 0:
            return None
        return id

    def get_neighbors(self, word: str = None, vector: np.ndarray = None, k: int = 10):
        if word is None and vector is None:
            raise ValueError("word of vector should be specified")
        if word is not None:
            vector = self._embeddings[self._vocab.get_id(word)]
        else:
            vector = self._normalize(vector)
        word_id = self._vocab.get_id(word)
        dist = cosine_dist(vector, self._embeddings)
        dist[word_id] = np.inf
        neighbor_idx = np.argpartition(dist, k)[:k]
        neighbor_dist = dist[neighbor_idx]
        sorted_idx = np.argsort(neighbor_dist)
        return [(self._vocab.get_word(neighbor_idx[i]), neighbor_dist[i]) for i in sorted_idx]
