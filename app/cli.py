"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import random

from app.core import (
    NgramLM,
    evaluate_model_and_baselines,
    load_corpus,
    milestone_2,
    milestone_3,
    to_sentences,
    to_tokens,
)


def _run_interactive(model: NgramLM, phrase: str, topk: int = 3) -> str:
    """Interactive step-by-step word acceptance/rejection loop."""
    tokens = to_tokens(phrase)
    print(f"\nStarting phrase: {' '.join(tokens)}")
    print("Commands: [1-9] pick suggestion | [c] type custom word | [d] done\n")

    while True:
        current = " ".join(tokens)
        suggestions = model.suggest_next(current, topk=topk)
        print(f"Current: {current}")
        for idx, (word, prob) in enumerate(suggestions, start=1):
            print(f"  {idx}. {word} (p={prob:.4f})")

        choice = input("Choose next word (number, 'c' for custom, 'd' to finish): ").strip().lower()
        if choice in ("d", "done", "q", "quit", ""):
            break
        if choice == "c":
            custom = input("Enter word: ").strip().lower()
            if custom:
                tokens.extend(to_tokens(custom))
        elif choice.isdigit() and 1 <= int(choice) <= len(suggestions):
            tokens.append(suggestions[int(choice) - 1][0])
        else:
            print("Invalid choice. Try again.")

    result = " ".join(tokens).strip()
    if result:
        result = result[0].upper() + result[1:]
        if not result.endswith((".", "!", "?")):
            result += "."
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="educational-content-expander",
        description="A CLI tool that expands educational notes and bullet points into complete, well-formed sentences using smoothed n-gram language models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Smoke-test command
    sub.add_parser("hello", help="check that everything is wired up")

    # Expand command
    p_expand = sub.add_parser("expand", help="expand a short phrase into a complete sentence")
    p_expand.add_argument("phrase", help="short phrase or note to expand")
    p_expand.add_argument("--n", type=int, default=3, help="n-gram order (default: 3)")
    p_expand.add_argument("--k", type=float, default=0.5, help="add-k smoothing parameter (default: 0.5)")
    p_expand.add_argument("--min-len", type=int, default=3, help="minimum number of generated words")
    p_expand.add_argument("--max-len", type=int, default=20, help="maximum number of generated words")
    p_expand.add_argument("--seed", type=int, default=0, help="random seed for sampling")
    p_expand.add_argument("--corpus", default=None, help="path to educational corpus .txt file")

    # Suggest command
    p_suggest = sub.add_parser("suggest", help="show top-k next-word suggestions for a phrase")
    p_suggest.add_argument("phrase", help="input phrase")
    p_suggest.add_argument("--topk", type=int, default=5, help="number of suggestions to display")
    p_suggest.add_argument("--n", type=int, default=3, help="n-gram order (default: 3)")
    p_suggest.add_argument("--k", type=float, default=0.5, help="add-k smoothing parameter (default: 0.5)")
    p_suggest.add_argument("--corpus", default=None, help="path to educational corpus .txt file")

    # Interactive mode
    p_inter = sub.add_parser("interactive", help="build a sentence step-by-step by accepting/rejecting suggestions")
    p_inter.add_argument("phrase", help="initial short phrase")
    p_inter.add_argument("--topk", type=int, default=3, help="suggestions shown per step")
    p_inter.add_argument("--n", type=int, default=3, help="n-gram order (default: 3)")
    p_inter.add_argument("--k", type=float, default=0.5, help="add-k smoothing parameter (default: 0.5)")
    p_inter.add_argument("--corpus", default=None, help="path to educational corpus .txt file")

    # Evaluate perplexity
    p_eval = sub.add_parser("evaluate", help="measure held-out perplexity on an 80/20 corpus split")
    p_eval.add_argument("--n", type=int, default=2, help="n-gram order (default: 2)")
    p_eval.add_argument("--k", type=float, default=0.5, help="add-k smoothing parameter (default: 0.5)")
    p_eval.add_argument("--corpus", default=None, help="path to educational corpus .txt file")
    p_eval.add_argument("--compare-all", action="store_true", help="compare uniform, unigram, bigram, and trigram models")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Educational Content Expander is ready.")
    elif args.command == "expand":
        expanded = milestone_3(
            phrase=args.phrase,
            corpus_path=args.corpus,
            n=args.n,
            k=args.k,
            seed=args.seed,
            min_len=args.min_len,
            max_len=args.max_len,
        )
        print(expanded)
    elif args.command == "suggest":
        model = milestone_2(corpus_path=args.corpus, n=args.n, k=args.k)
        suggestions = model.suggest_next(args.phrase, topk=args.topk)
        print(f"Next-word suggestions for '{args.phrase}':")
        for word, prob in suggestions:
            print(f"  - {word:<15} (p={prob:.4f})")
    elif args.command == "interactive":
        model = milestone_2(corpus_path=args.corpus, n=args.n, k=args.k)
        final_sentence = _run_interactive(model, args.phrase, topk=args.topk)
        print(f"\nFinal sentence: {final_sentence}")
    elif args.command == "evaluate":
        raw = load_corpus(args.corpus)
        sents = to_sentences(raw)
        rng = random.Random(0)
        rng.shuffle(sents)
        cut = max(1, int(0.8 * len(sents)))
        train_text = ". ".join(sents[:cut]) + "."
        test_text = ". ".join(sents[cut:]) + "."

        res = evaluate_model_and_baselines(raw, n=args.n, k=args.k, train_ratio=0.8, seed=0)
        ngram_label = f"N-gram Model (n={args.n}, k={args.k})"
        red_label = f"{res['held_out_reduction_pct']:.1f}%"

        print("=" * 68)
        print("Educational Content Expander - Perplexity & Baseline Evaluation")
        print("=" * 68)
        print(f"Corpus: {res['sentences_total']} sentences ({res['sentences_train']} train / {res['sentences_test']} test, 80/20 split)")
        print(f"Vocabulary Size (|V|): {res['vocab_size']} unique tokens (including </s>)")
        print("-" * 68)
        print(f"{'Model / Baseline':<28} {'Train PPL':>12} {'Held-out PPL':>14} {'Reduction':>10}")
        print("-" * 68)
        print(f"{'Uniform Random Baseline':<28} {res['uniform_ppl']:>12.2f} {res['uniform_ppl']:>14.2f} {'(base)':>10}")
        print(f"{'Unigram Baseline (n=1)':<28} {res['unigram_train_ppl']:>12.2f} {res['unigram_test_ppl']:>14.2f} {'0.0%':>10}")
        print(f"{ngram_label:<28} {res['ngram_train_ppl']:>12.2f} {res['ngram_test_ppl']:>14.2f} {red_label:>10}")

        if args.compare_all:
            for order in [2, 3, 4]:
                if order == args.n:
                    continue
                m = NgramLM(n=order, k=args.k).fit(train_text)
                tr_ppl = m.perplexity(train_text)
                te_ppl = m.perplexity(test_text)
                order_label = f"N-gram Model (n={order}, k={args.k})"
                red = ((res['unigram_test_ppl'] - te_ppl) / res['unigram_test_ppl']) * 100
                red_order_label = f"{red:.1f}%"
                print(f"{order_label:<28} {tr_ppl:>12.2f} {te_ppl:>14.2f} {red_order_label:>10}")

        print("-" * 68)
        print(f"Train perplexity    (n={args.n}, k={args.k}): {res['ngram_train_ppl']:.2f}")
        print(f"Held-out perplexity (n={args.n}, k={args.k}): {res['ngram_test_ppl']:.2f}")
        print("=" * 68)


if __name__ == "__main__":
    main()
