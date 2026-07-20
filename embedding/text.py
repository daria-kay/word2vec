import random
import re
from itertools import accumulate
from collections import Counter

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


class Word2VecParser:

    def __init__(self, corpora_path: str):
        self._corpora_path = corpora_path
        self._vocab = dict()
        self._vocab["UNK"] = 0
        self._reverse_vocab = []
        self._word_pattern = re.compile(r'[A-Za-z0-9_\'-]+')
        self._sentence_pattern = re.compile(r'[^.!?]+')
        self._words_freq = None

    def _raw_word_iter(self):
        with open(self._corpora_path, 'r') as corpora:
            for line in corpora:
                for sentence_match in self._sentence_pattern.finditer(line.lower()):
                    for word_match in self._word_pattern.finditer(sentence_match[0]):
                        yield word_match[0]

    def _build_vocab(self, min_count: int) -> Vocab:
        if self._words_freq is None:
            self._words_freq = Counter()
            for word in self._raw_word_iter():
                self._words_freq[word] += 1
        vocab = Vocab()
        for word, freq in self._words_freq.items():
            if freq < min_count:
                continue
            vocab.add(word, freq)

        return vocab

    def _context_iter(self, vocab, window_size):
        window_half = window_size // 2
        with open(self._corpora_path, 'r') as corpora:
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

    def parse_positive_negative_samples(self,
                                        output_path: str,
                                        k: int = 5,
                                        window_size: int = 5,
                                        min_count: int = 10,
                                        ):
        vocab = self._build_vocab(min_count)
        cum_weights = list(accumulate([self._words_freq[w] ** 0.75 for w in vocab.words]))
        with open(output_path, 'w') as output_file:
            for word, context_words in tqdm.tqdm(self._context_iter(vocab, window_size)):
                cum_weights_copy = cum_weights.copy()
                for context_word in context_words:
                    print(word, context_word, 1, sep=',', file=output_file)
                    context_word_id = vocab.get_id(context_word)
                    cum_weights_copy[context_word_id] = cum_weights_copy[context_word_id - 1]
                for negative_sample in random.choices(vocab.words, cum_weights=cum_weights_copy, k=k):
                    print(word, negative_sample, 0, sep=',', file=output_file)
        return vocab
