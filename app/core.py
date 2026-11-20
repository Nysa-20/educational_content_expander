"""Core n-gram language model and educational text expansion logic."""
from __future__ import annotations

import re
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
