import os
import pickle
import random
import re
from collections import Counter
from itertools import accumulate
from pathlib import Path
from typing import Tuple

import tqdm


class Vocab:

    def __init__(self):
        self._vocab = dict()
        self._reverse_vocab = ["UNK"]
        self._vocab["UNK"] = 0
        self._counter = Counter()

    @property
    def words(self):
        return self._reverse_vocab

    def add(self, word, frequency):
        if word in self._vocab:
            return
        self._vocab[word] = len(self._vocab)
        self._reverse_vocab.append(word)
        self._counter[word] = frequency

    def get_id(self, word) -> int:
        return self._vocab.get(word, 0)

    def get_word(self, id):
        if id >= len(self._reverse_vocab):
            return self._reverse_vocab[0]
        return self._reverse_vocab[id]

    def top_n(self, n):
        return self._counter.most_common(n)

    def __len__(self):
        return len(self._reverse_vocab)

    def __contains__(self, item):
        return item in self._vocab


class Word2VecDatasetBuilder:
    _DATASET_FILENAME: str = 'dataset.csv'
    _VOCAB_FILENAME: str = 'vocab.pkl'

    def __init__(self, word_pattern: re.Pattern | None = None, sentence_pattern: re.Pattern | None = None, seed: int = 42):
        self._word_pattern = word_pattern
        self._sentence_pattern = sentence_pattern
        if self._word_pattern is None:
            self._word_pattern = re.compile(r'[A-Za-z0-9_\'-]+')
        if self._sentence_pattern is None:
            self._sentence_pattern = re.compile(r'[^.!?]+')
        self._random = random.Random(seed)

    def _raw_word_iter(self, corpora_path: str):
        with open(corpora_path, 'r') as corpora:
            for line in corpora:
                for sentence_match in self._sentence_pattern.finditer(line.lower()):
                    for word_match in self._word_pattern.finditer(sentence_match[0]):
                        yield word_match[0]

    def _build_vocab(self, corpora_path: str, min_count: int) -> Tuple[Vocab, Counter]:
        words_freq = Counter()
        for word in self._raw_word_iter(corpora_path):
            words_freq[word] += 1
        vocab = Vocab()
        for word, freq in words_freq.items():
            if freq < min_count:
                continue
            vocab.add(word, freq)

        return vocab, words_freq

    def _context_iter(self, corpora_path: str, vocab: Vocab, window_size: int):
        window_half = window_size // 2
        with open(corpora_path, 'r') as corpora:
            for line in corpora:
                for sentence_match in self._sentence_pattern.finditer(line.lower()):
                    sentence = [
                        word_match[0]
                        for word_match in self._word_pattern.finditer(sentence_match[0]) if word_match[0] in vocab
                    ]
                    for center_idx in range(len(sentence)):
                        center_word = sentence[center_idx]
                        left = max(0, center_idx - window_half)
                        right = min(len(sentence), center_idx + window_half + 1)
                        context_words = set(sentence[left:right])
                        context_words.remove(center_word)
                        yield center_word, context_words

    def build_dataset(self,
                      corpora_path: str,
                      output_path: str,
                      k: int = 20,
                      window_size: int = 5,
                      min_count: int = 10,
                      ):
        os.makedirs(output_path, exist_ok=True)
        vocab, words_freq = self._build_vocab(corpora_path, min_count)
        cum_weights = list(accumulate([words_freq[w] ** 0.75 for w in vocab.words]))
        with open(Path(output_path, self._DATASET_FILENAME), 'w') as output_file:
            for word, context_words in tqdm.tqdm(self._context_iter(corpora_path, vocab, window_size)):
                cum_weights_copy = cum_weights.copy()
                for context_word in context_words:
                    print(word, context_word, 1, sep=',', file=output_file)
                    context_word_id = vocab.get_id(context_word)
                    cum_weights_copy[context_word_id] = cum_weights_copy[context_word_id - 1]
                for negative_sample in self._random.choices(vocab.words, cum_weights=cum_weights_copy, k=k):
                    print(word, negative_sample, 0, sep=',', file=output_file)
        with open(Path(output_path, self._VOCAB_FILENAME), 'wb') as vocab_file:
            pickle.dump(vocab, file=vocab_file)
