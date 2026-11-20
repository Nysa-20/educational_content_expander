"""Smoke tests verifying package imports, baseline models, and CLI command execution."""
from app.cli import main
from app.core import evaluate_model_and_baselines, load_corpus, milestone_1, milestone_2


def test_package_smoke():
    """Verify that the corpus and n-gram models initialize properly."""
    stats = milestone_1()
    assert len(stats["sentences"]) > 0
    assert len(stats["vocabulary"]) > 0
    model = milestone_2(n=2, k=0.5)
    assert model.prob("energy", "cannot") > 0.0


def test_cli_hello_runs(capsys):
    """Verify that `python -m app.cli hello` runs."""
    main(["hello"])
    out = capsys.readouterr().out
    assert "Educational Content Expander is ready" in out


def test_cli_expand_runs(capsys):
    """Verify that `expand` generates text without error."""
    main(["expand", "photosynthesis is", "--n", "2", "--max-len", "5"])
    out = capsys.readouterr().out.strip()
    assert out.startswith("Photosynthesis is")
    assert out.endswith(".")


def test_cli_suggest_runs(capsys):
    """Verify that `suggest` returns candidate words."""
    main(["suggest", "energy", "--topk", "3"])
    out = capsys.readouterr().out.strip()
    assert "Next-word suggestions" in out
    assert "energy" in out


def test_cli_evaluate_runs(capsys):
    """Verify that `evaluate` prints perplexity and baseline comparison."""
    main(["evaluate", "--n", "2", "--k", "0.5"])
    out = capsys.readouterr().out
    assert "Perplexity & Baseline Evaluation" in out
    assert "Train perplexity" in out
    assert "Held-out perplexity" in out
    assert "Uniform Random Baseline" in out
    assert "Unigram Baseline" in out


def test_cli_evaluate_compare_all_runs(capsys):
    """Verify that `evaluate --compare-all` runs across model orders."""
    main(["evaluate", "--n", "2", "--k", "0.5", "--compare-all"])
    out = capsys.readouterr().out
    assert "n=3" in out
    assert "n=4" in out
