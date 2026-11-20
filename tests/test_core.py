"""Tests for Milestones 1–4, baseline comparisons, smoothing, and CLI commands."""
import pytest
from app.core import (
    NgramLM,
    evaluate_model_and_baselines,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
)


def test_milestone_1(tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Photosynthesis converts sunlight into chemical energy. "
        "Plants absorb carbon dioxide and water.",
        encoding="utf-8",
    )
    data = milestone_1(sample)
    assert len(data["sentences"]) == 2
    assert "photosynthesis" in data["vocabulary"]
    assert len(data["tokens"]) == 12


def test_milestone_2(tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Energy flows through ecosystems. "
        "Energy transforms from potential energy into kinetic energy.",
        encoding="utf-8",
    )
    model = milestone_2(corpus_path=sample, n=2, k=0.5)
    assert isinstance(model, NgramLM)
    # Unseen pair must still receive strictly positive probability under add-k smoothing
    unseen_prob = model.prob("ecosystems", "quantum")
    seen_prob = model.prob("kinetic", "energy")
    assert unseen_prob > 0.0
    assert seen_prob > unseen_prob
    assert model.perplexity("Energy flows through ecosystems.") > 1.0


def test_milestone_3(tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Cellular respiration breaks down glucose molecules to release usable energy. "
        "Cellular respiration powers essential metabolic functions inside living cells.",
        encoding="utf-8",
    )
    expanded = milestone_3(
        "cellular respiration",
        corpus_path=sample,
        n=3,
        k=0.5,
        seed=1,
        min_len=3,
        max_len=12,
    )
    assert expanded.startswith("Cellular respiration")
    assert expanded.endswith(".")
    assert len(expanded.split()) >= 5


def test_milestone_4(capsys, tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Natural selection drives evolutionary adaptation across generations.",
        encoding="utf-8",
    )
    milestone_4(["expand", "natural selection", "--corpus", str(sample), "--min-len", "2", "--max-len", "8"])
    out = capsys.readouterr().out.strip()
    assert out.startswith("Natural selection")
    assert out.endswith(".")


def test_baseline_comparison(tmp_path):
    """Verify that n-gram models achieve lower perplexity than unigram and uniform baselines."""
    sample = tmp_path / "edu_large.txt"
    sample.write_text(
        "Photosynthesis converts sunlight into chemical energy. "
        "During photosynthesis green plants produce glucose and oxygen. "
        "Cellular respiration breaks down glucose molecules to release energy. "
        "Living cells use cellular respiration to power metabolic processes. "
        "Ecosystems rely on photosynthesis and respiration to sustain life. "
        "Energy transfers from producers to primary consumers in food webs. "
        "Decomposers break down organic matter and return nutrients to soil. "
        "Biodiversity supports ecosystem resilience against environmental disturbances.",
        encoding="utf-8",
    )
    eval_res = evaluate_model_and_baselines(str(sample), n=2, k=0.5, train_ratio=0.75, seed=42)
    # Uniform baseline perplexity equals vocabulary size
    assert eval_res["uniform_ppl"] == float(eval_res["vocab_size"])
    # Bigram train perplexity must be significantly lower than unigram train perplexity
    assert eval_res["ngram_train_ppl"] < eval_res["unigram_train_ppl"]
    # Bigram train perplexity must be significantly lower than uniform baseline
    assert eval_res["ngram_train_ppl"] < eval_res["uniform_ppl"]


def test_ngram_parameter_validation():
    """Verify that invalid n and k raise ValueError."""
    with pytest.raises(ValueError, match="n must be at least 1"):
        NgramLM(n=0)
    with pytest.raises(ValueError, match="k must be positive"):
        NgramLM(n=2, k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        NgramLM(n=2, k=-0.5)


def test_suggestion_ordering(tmp_path):
    """Verify that suggest_next ranks candidates in descending order of probability."""
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Green plants convert light into chemical energy. "
        "Green plants convert water and nutrients into tissue. "
        "Green plants absorb sunlight for growth.",
        encoding="utf-8",
    )
    model = NgramLM(n=3, k=0.5).fit(sample.read_text(encoding="utf-8"))
    suggestions = model.suggest_next("green plants", topk=3)
    assert len(suggestions) >= 1
    probs = [p for _, p in suggestions]
    assert probs == sorted(probs, reverse=True)
    assert suggestions[0][0] == "convert"
