import math
import re
from collections import Counter


# standard BM25 hyperparameters - these are the typical defaults
# k1 controls term frequency saturation, b controls length normalization
# I kept these at the classic values, didn't tune them
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer, lowercase everything.
    Nothing fancy - just good enough for keyword matching on research papers.
    stopwords removed manually since we can't use nltk here.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return tokens


class BM25Index:
    """
    BM25 index built on top of a list of text chunks.

    I implemented this from scratch - the formula is:
        score(d, q) = sum over query terms t of:
            IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1 - b + b*dl/avgdl))

    where:
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        tf(t,d) = term frequency of t in document d
        dl = document length (number of tokens)
        avgdl = average document length across all docs
    """

    def __init__(self, corpus: list[str]):
        self.corpus_size = len(corpus)
        self.tokenized = [tokenize(doc) for doc in corpus]

        # document lengths
        self.doc_lengths = [len(tokens) for tokens in self.tokenized]
        self.avgdl = sum(self.doc_lengths) / max(self.corpus_size, 1)

        # term -> document frequency (how many docs contain this term)
        self.df: dict[str, int] = {}
        for tokens in self.tokenized:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        # precompute IDF for every term we've seen
        self.idf: dict[str, float] = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

        # term frequency per document - store as list of Counters
        self.tf_per_doc = [Counter(tokens) for tokens in self.tokenized]

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for a single (query, document) pair."""
        query_terms = tokenize(query)
        dl = self.doc_lengths[doc_idx]
        tf_map = self.tf_per_doc[doc_idx]

        total = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            numerator = tf * (K1 + 1)
            denominator = tf + K1 * (1 - B + B * dl / max(self.avgdl, 1))
            total += self.idf[term] * numerator / denominator

        return total

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Return (doc_idx, score) pairs sorted by score descending.
        Only returns docs with score > 0 - no point including irrelevant ones.
        """
        scores = []
        for i in range(self.corpus_size):
            s = self.score(query, i)
            if s > 0:
                scores.append((i, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# pretty standard English stopwords list - just hardcoded it since we can't use nltk
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "we", "our", "they", "their", "you",
    "your", "he", "she", "his", "her", "i", "my", "me", "us", "as", "if",
    "not", "no", "so", "also", "than", "then", "when", "where", "which",
    "who", "what", "how", "all", "each", "both", "more", "most", "other",
    "into", "through", "during", "before", "after", "above", "between",
    "such", "while", "about", "up", "out", "any", "only", "same", "over",
}
