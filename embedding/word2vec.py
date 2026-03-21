import random
import sys
from typing import Iterable

import numpy as np
from tqdm.notebook import tqdm

from nn import Adam, NegativeSamplingLoss, SkipGram
from .tokenizer import Tokenizer


class Word2VecDataLoader(Iterable):

    def __init__(self, dataset_file: str, tokenizer: Tokenizer, window_size: int = 5, batch_size: int = 256,
                 k: int = 5):
        self.filename = dataset_file
        self.tokenizer = tokenizer
        self.window_half = window_size // 2
        self.batch_size = batch_size
        self.k = k

    def __iter__(self):
        central_words, samples, labels = [], [], []
        with open(self.filename, "r") as file:
            for line in file:
                words = line.split()
                n = len(words)
                for i in range(n):
                    for j in range(max(0, i - self.window_half), min(n, i + self.window_half + 1)):
                        if i == j:
                            continue

                        central = self.tokenizer.tokenize(words[i])
                        positive = self.tokenizer.tokenize(words[j])

                        negative_samples = []
                        while len(negative_samples) < self.k:
                            neg = random.randint(0, self.tokenizer.vocab_size - 1)
                            if neg != positive and neg != central:
                                negative_samples.append(neg)

                        central_words.append(central)
                        negative_samples.append(positive)
                        samples.append(np.array(negative_samples, dtype=np.int32))
                        label = np.zeros((self.k + 1,))
                        label[-1] = 1
                        labels.append(label)

                        if len(central_words) == self.batch_size:
                            yield np.array(central_words, dtype=np.int32), np.stack(samples), np.stack(labels)
                            central_words, samples, labels = [], [], []

        if len(central_words) > 0:
            yield np.array(central_words, dtype=np.int32), np.stack(samples), np.stack(labels)


class Word2Vec:

    def __init__(self,
                 train_filename: str,
                 val_filename: str,
                 embedding_size: int,
                 tokenizer: Tokenizer,
                 window_size: int = 5,
                 batch_size: int = 256
                 ):
        self.train_data = Word2VecDataLoader(train_filename, tokenizer, window_size=window_size, batch_size=batch_size)
        self.val_data = Word2VecDataLoader(val_filename, tokenizer, window_size=window_size, batch_size=batch_size)
        self.model = SkipGram(tokenizer.vocab_size, embedding_size)
        self.embedding_size = embedding_size
        self.objective = NegativeSamplingLoss()

    def train(self, n_iter: int = 5, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        optimizer = Adam(self.model, lr=lr, beta1=beta1, beta2=beta2)
        train_losses = []
        val_losses = []

        for _ in tqdm(range(n_iter), "epoch", file=sys.stdout, position=0):
            train_loss = 0.0
            train_samples = 0
            for central_word, context_words, labels in self.train_data:
                optimizer.zero_grad()
                logits = self.model(central_word, context_words)

                loss = self.objective.forward(logits, labels)
                train_loss += loss.sum()
                train_samples += central_word.shape[0]

                self.model.backward(central_word, context_words, self.objective.backward(logits, labels))
                optimizer.step()
            train_losses.append(train_loss / train_samples)
            print(f"Train loss: {train_losses[-1]}")

            val_loss = 0.0
            val_samples = 0
            for central_word, context_words, labels in self.val_data:
                logits = self.model(central_word, context_words)
                loss = self.objective.forward(logits, labels)
                val_loss += loss.sum()
                val_samples += central_word.shape[0]
            val_losses.append(val_loss / val_samples)
            print(f"Val loss: {val_losses[-1]}")

        return train_losses, val_losses

    def get(self, word: str):
        idx = self.tokenizer.tokenize(word)
        return self.model.central_embeddings[idx, :]
