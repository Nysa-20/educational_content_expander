"""Smoke tests verifying package imports and basic CLI execution."""
from app.cli import main
from app.core import milestone_1, milestone_2


def test_package_smoke():
    stats = milestone_1()
    assert len(stats["sentences"]) > 0
    assert len(stats["vocabulary"]) > 0
    model = milestone_2(n=2, k=0.5)
    assert model.prob("energy", "cannot") > 0.0


def test_cli_hello_runs(capsys):
    main(["hello"])
    assert "is ready" in capsys.readouterr().out


def test_cli_expand_runs(capsys):
    main(["expand", "photosynthesis is", "--n", "2", "--max-len", "5"])
    out = capsys.readouterr().out.strip()
    assert out.startswith("Photosynthesis is")
    assert out.endswith(".")


def test_cli_suggest_runs(capsys):
    main(["suggest", "energy", "--topk", "2"])
    out = capsys.readouterr().out.strip()
    assert "Next-word suggestions" in out


def test_cli_evaluate_runs(capsys):
    main(["evaluate", "--n", "2", "--k", "0.5"])
    out = capsys.readouterr().out
    assert "Train perplexity" in out
    assert "Held-out perplexity" in out
