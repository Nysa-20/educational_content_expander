"""Core n-gram language model and educational text expansion logic."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CORPUS_PATH = Path("data/corpus.txt")

FALLBACK_CORPUS = """
Photosynthesis is the biological process by which green plants convert light energy into chemical energy.
Cellular respiration breaks down glucose molecules to release usable energy in the form of adenosine triphosphate.
Mitosis is a type of cell division that produces two genetically identical daughter cells for growth and repair.
Natural selection favors individuals with advantageous traits that improve survival and reproductive success.
Newton's first law of motion states that an object remains at rest or in uniform motion unless acted upon by an external force.
The law of conservation of energy states that energy cannot be created or destroyed, only transformed from one form to another.
Scientific inquiry relies on testable hypotheses, controlled experiments, empirical measurement, and peer review.
""".strip()


def load_corpus(path: str | Path | None = None) -> str:
    """Load educational corpus text from disk, falling back to built-in educational sentences."""
    target = Path(path) if path else DEFAULT_CORPUS_PATH
    if target.exists():
        return target.read_text(encoding="utf-8")
    return FALLBACK_CORPUS


def to_sentences(text: str) -> list[str]:
    """Split raw text into cleaned sentences."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def to_tokens(sentence: str) -> list[str]:
    """Lowercase and tokenize a sentence into word tokens."""
    return re.findall(r"[a-z']+", sentence.lower())


def build(sents: list[str], n: int) -> defaultdict[tuple[str, ...], Counter[str]]:
    """Build n-gram continuation counts: context (n-1 words) -> Counter of next words."""
    if n < 1:
        raise ValueError("n must be at least 1")
    foll: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for s in sents:
        tokens = to_tokens(s)
        if not tokens:
            continue
        toks = ["<s>"] * (n - 1) + tokens + ["</s>"]
        for i in range(len(toks) - n + 1):
            ctx = tuple(toks[i : i + n - 1])
            nxt = toks[i + n - 1]
            foll[ctx][nxt] += 1
    return foll


class NgramLM:
    """A smoothed n-gram language model with add-k smoothing and perplexity calculation."""

    def __init__(self, n: int = 3, k: float = 0.5):
        if n < 1:
            raise ValueError("n must be at least 1")
        if k <= 0:
            raise ValueError("k must be positive for add-k smoothing")
        self.n = n
        self.k = k
        self.sents: list[str] = []
        self.foll: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.V: set[str] = {"</s>"}

    def fit(self, text: str) -> NgramLM:
        """Fit n-gram counts and vocabulary from raw corpus text."""
        self.sents = to_sentences(text)
        self.foll = build(self.sents, self.n)
        self.V = {t for s in self.sents for t in to_tokens(s)} | {"</s>"}
        return self

    def prob(self, ctx: tuple[str, ...] | str, word: str) -> float:
        """Compute add-k smoothed probability P(word | ctx)."""
        if self.n == 1:
            ctx_tuple: tuple[str, ...] = ()
        elif isinstance(ctx, str):
            ctx_tuple = tuple((["<s>"] * (self.n - 1) + to_tokens(ctx))[-(self.n - 1) :])
        else:
            ctx_tuple = tuple((("<s>",) * (self.n - 1) + ctx)[-(self.n - 1) :])
        c = self.foll.get(ctx_tuple, Counter())
        total = sum(c.values())
        return (c[word.lower()] + self.k) / (total + self.k * len(self.V))

    def perplexity(self, text: str) -> float:
        """Compute cross-entropy perplexity on held-out text."""
        log_sum = 0.0
        total_steps = 0
        for s in to_sentences(text):
            tokens = to_tokens(s)
            if not tokens:
                continue
            toks = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for i in range(len(toks) - self.n + 1):
                ctx = tuple(toks[i : i + self.n - 1])
                w = toks[i + self.n - 1]
                log_sum += math.log(self.prob(ctx, w))
                total_steps += 1
        if total_steps == 0:
            raise ValueError("Cannot compute perplexity on empty text.")
        return math.exp(-log_sum / total_steps)


# --- Milestone 1: Gather a corpus of educational texts relevant to a subject.
def milestone_1(corpus_path: str | Path | None = None) -> dict:
    """Load and preprocess the educational corpus, returning summary statistics."""
    raw = load_corpus(corpus_path)
    sentences = to_sentences(raw)
    tokens = [t for s in sentences for t in to_tokens(s)]
    return {
        "sentences": sentences,
        "tokens": tokens,
        "vocabulary": sorted(set(tokens)),
    }


# --- Milestone 2: Build the n-gram model, including add-k smoothing.
def milestone_2(corpus_path: str | Path | None = None, n: int = 3, k: float = 0.5) -> NgramLM:
    """Fit and return a smoothed n-gram language model on the educational corpus."""
    raw = load_corpus(corpus_path)
    return NgramLM(n=n, k=k).fit(raw)
