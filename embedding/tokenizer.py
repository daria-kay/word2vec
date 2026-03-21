import re

class Tokenizer:

    def __init__(self):
        self.chunk_size = 100_000
        self.sentence_seps = {".", "!", "?", ";"}
        self.vocab = dict()
        self.vocab["UNK"] = 0
        self.reverse_vocab = []
        self.vocab_size = 1
        self.sentence_count = 0
        self.token_pattern = re.compile(r"[A-Za-z]+|[^\w\s]")

    def _read_chunks(self, filename: str):
        chunk = []
        with open(filename, "r") as file:
            for line in file:
                line = line.rstrip("\n")
                chunk.append(line)
                if len(chunk) >= self.chunk_size:
                    yield chunk
                    chunk = []

            if chunk:
                yield chunk

    def _lemmatize(self, text_chunk: list[str]):
        sentence = []
        for text in text_chunk:
            for token in self.token_pattern.findall(text):

                if re.fullmatch(r"[A-Za-z]+", token):
                    sentence.append(token.lower())
                    continue

                if token in self.sentence_seps:
                    if len(sentence) > 1:
                        yield sentence
                        sentence = []
                    else:
                        sentence = []
                    continue

            if len(sentence) > 1:
                yield sentence
                sentence = []
            else:
                sentence = []

        if len(sentence) > 1:
            yield sentence

    def fit(self, input_filename: str, output_filename: str, sentences_count: int = None):
        stop = False
        with open(output_filename, "w", encoding="utf-8") as output:
            for chunk in self._read_chunks(input_filename):
                for sentence in self._lemmatize(chunk):
                    for word in sentence:
                        if word not in self.vocab:
                            self.vocab[word] = self.vocab_size
                            self.vocab_size += 1
                    print(" ".join(sentence), file=output)
                    self.sentence_count += 1
                    if sentences_count and self.sentence_count >= sentences_count:
                        stop = True
                        break
                if stop:
                    break

        self.reverse_vocab = [None] * self.vocab_size
        for word, idx in self.vocab.items():
            self.reverse_vocab[idx] = word

        print(f"Vocab len: {self.vocab_size}")

    def tokenize(self, word: str) -> int:
        if word in self.vocab:
            return self.vocab[word]
        return self.vocab["UNK"]