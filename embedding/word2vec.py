import os
import pickle
from typing import Iterable

import numpy as np
import polars as pl
from tqdm.notebook import tqdm

from nn import Adam, BinaryClassificationLoss, SkipGram, Node
from . import Vocab


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
    _VOCAB_FILENAME: str = 'tokenizer.pkl'
    _EMBEDDINGS_FILENAME: str = 'embeddings.pkl'

    def __init__(self,
                 embedding_size: int,
                 vocab: Vocab,
                 train_data: Word2VecDataloader | None = None,
                 val_data: Word2VecDataloader | None = None,
                 skip_gram: SkipGram | None = None,
                 objective: Node | None = None,
                 pretrained_embeddings: np.ndarray | None = None,
                 seed: int = 42
                 ):
        self.embedding_size = embedding_size
        self.vocab = vocab

        self._train_data = train_data
        self._val_data = val_data
        self._skip_gram = skip_gram
        self._objective = objective
        self._embeddings = pretrained_embeddings
        self._seed = seed

        self.train_losses_ = None
        self.val_losses_ = None
        self.gradient_norms_ = None
        self.parameter_norms_ = None

    @classmethod
    def from_scratch(cls,
                     dataset: pl.LazyFrame,
                     vocab: Vocab,
                     embedding_size: int,
                     batch_size: int = 10_000,
                     val_percent: int = 10,
                     ):
        dataset = dataset.with_row_index()
        val_data = Word2VecDataloader(
            dataset.filter(pl.first() % val_percent == 0),
            vocab,
            batch_size
        )
        train_data = Word2VecDataloader(
            dataset.filter(pl.first() % val_percent != 0),
            vocab,
            batch_size
        )
        skip_gram = SkipGram(len(vocab), embedding_size)
        objective = BinaryClassificationLoss()
        return cls(embedding_size, vocab, train_data=train_data, val_data=val_data, skip_gram=skip_gram,
                   objective=objective)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        with open(pretrained_model_path + '/' + cls._VOCAB_FILENAME, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)
        with open(pretrained_model_path + '/' + cls._EMBEDDINGS_FILENAME, 'rb') as embeddings_file:
            embeddings = pickle.load(embeddings_file)
        return cls(embeddings.shape[-1], tokenizer, pretrained_embeddings=embeddings)

    def train(self, epoches: int = 1, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        if self._train_data is None or self._val_data is None:
            raise ValueError('train data must be specified')

        optimizer = Adam(self._skip_gram, lr=lr, beta1=beta1, beta2=beta2)

        train_losses = []
        val_losses = []

        for _ in tqdm(range(epoches), "epoch"):
            train_loss = 0.0
            train_samples = 0
            for central_word, context_words, labels in self._train_data:
                optimizer.zero_grad()
                logits = self._skip_gram(central_word, context_words)

                loss = self._objective.forward(logits, labels)
                train_loss += loss.sum()
                train_samples += central_word.shape[0]

                self._skip_gram.backward(central_word, context_words, self._objective.backward(logits, labels))
                optimizer.step()
            train_losses.append(train_loss / train_samples)

            val_loss = 0.0
            val_samples = 0
            for central_word, context_words, labels in self._val_data:
                logits = self._skip_gram(central_word, context_words)
                loss = self._objective.forward(logits, labels)
                val_loss += loss.sum()
                val_samples += central_word.shape[0]
            val_losses.append(val_loss / val_samples)

        self.train_losses_ = train_losses
        self.val_losses_ = val_losses
        self.gradient_norms_ = optimizer.gradient_norms
        self.parameter_norms_ = optimizer.parameter_norms

        self._embeddings = self._skip_gram.central_embeddings.weights.copy()

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(save_directory + '/' + self._VOCAB_FILENAME, 'wb') as vocab_file:
            pickle.dump(self.vocab, vocab_file)
        with open(save_directory + '/' + self._EMBEDDINGS_FILENAME, 'wb') as embeddings_file:
            pickle.dump(self._embeddings, embeddings_file)

    def __getitem__(self, word):
        if self._embeddings is None:
            raise ValueError('model should be trained or initialized from pretrained embeddings')
        return self._embeddings[self.vocab.get_id(word)]

    def __contains__(self, word):
        return word in self.vocab
