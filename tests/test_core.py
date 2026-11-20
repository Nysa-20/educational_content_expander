"""Tests for Milestones 1–2 of Educational Content Expander."""
import pytest
from app.core import NgramLM, milestone_1, milestone_2


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


def test_ngram_parameter_validation():
    """Verify that invalid n and k raise ValueError."""
    with pytest.raises(ValueError, match="n must be at least 1"):
        NgramLM(n=0)
    with pytest.raises(ValueError, match="k must be positive"):
        NgramLM(n=2, k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        NgramLM(n=2, k=-0.5)
