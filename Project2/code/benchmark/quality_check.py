"""
Lightweight output quality check (addresses professor's concern).

Compares baseline vs EAGLE-3 outputs on the same prompts to confirm
speculative decoding does NOT degrade output quality.

Checks:
  - Summarization : ROUGE-L score (baseline vs eagle3 outputs)
  - Code          : syntax validity via ast.parse()
  - Chat          : prints 5 side-by-side samples for manual review

Usage:
    python benchmark/quality_check.py \
        --baseline-url http://localhost:8000 \
        --eagle3-url http://localhost:8001 \
        --task summarization \
        --prompts-file data/prompts/summarization_50.jsonl \
        --n-check 10
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import requests
from rouge_score import rouge_scorer

MAX_TOKENS = 256
TIMEOUT = 60


def _generate(server_url: str, model_name: str, prompt: str) -> str:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": False,
    }
    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def check_summarization(
    baseline_url: str,
    eagle3_url: str,
    prompts: list[dict],
    n: int,
) -> None:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sample = prompts[:n]

    print(f"\n[Summarization Quality] Checking {n} prompts...\n")
    scores = []

    for i, p in enumerate(sample):
        out_baseline = _generate(baseline_url, "baseline", p["prompt"])
        out_eagle3 = _generate(eagle3_url, "eagle3", p["prompt"])

        # ROUGE-L between baseline and eagle3 outputs (not reference)
        # We want them to be similar to each other → high score means consistent
        score = scorer.score(out_baseline, out_eagle3)["rougeL"].fmeasure
        scores.append(score)
        print(f"  [{i+1:02d}] ROUGE-L(baseline vs eagle3) = {score:.3f}")

    avg = sum(scores) / len(scores)
    print(f"\n  Average ROUGE-L: {avg:.3f}")
    if avg >= 0.70:
        print("  RESULT: PASS — outputs are highly consistent (>=0.70)")
    elif avg >= 0.50:
        print("  RESULT: WARN — outputs differ somewhat (0.50–0.70), inspect samples")
    else:
        print("  RESULT: FAIL — significant output divergence (<0.50)")


def check_code(
    baseline_url: str,
    eagle3_url: str,
    prompts: list[dict],
    n: int,
) -> None:
    sample = prompts[:n]
    print(f"\n[Code Quality] Checking syntax validity for {n} prompts...\n")

    baseline_pass = 0
    eagle3_pass = 0

    for i, p in enumerate(sample):
        out_baseline = _generate(baseline_url, "baseline", p["prompt"])
        out_eagle3 = _generate(eagle3_url, "eagle3", p["prompt"])

        # Extract code block if model wraps in markdown
        def extract_code(text: str) -> str:
            if "```python" in text:
                return text.split("```python")[1].split("```")[0]
            if "```" in text:
                return text.split("```")[1].split("```")[0]
            return text

        code_b = extract_code(out_baseline)
        code_e = extract_code(out_eagle3)

        b_ok = _is_valid_python(code_b)
        e_ok = _is_valid_python(code_e)

        if b_ok:
            baseline_pass += 1
        if e_ok:
            eagle3_pass += 1

        status = "OK" if b_ok == e_ok else "DIFF"
        print(f"  [{i+1:02d}] baseline={'OK' if b_ok else 'FAIL'}  eagle3={'OK' if e_ok else 'FAIL'}  [{status}]")

    print(f"\n  Baseline syntax pass : {baseline_pass}/{n}")
    print(f"  EAGLE-3 syntax pass  : {eagle3_pass}/{n}")
    if eagle3_pass >= baseline_pass:
        print("  RESULT: PASS — EAGLE-3 code quality matches or exceeds baseline")
    else:
        print(f"  RESULT: WARN — EAGLE-3 has {baseline_pass - eagle3_pass} more syntax failures")


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def check_chat(
    baseline_url: str,
    eagle3_url: str,
    prompts: list[dict],
    n: int,
) -> None:
    sample = prompts[:n]
    print(f"\n[Chat Quality] Side-by-side output comparison ({n} samples)\n")
    print("=" * 80)

    for i, p in enumerate(sample):
        out_baseline = _generate(baseline_url, "baseline", p["prompt"])
        out_eagle3 = _generate(eagle3_url, "eagle3", p["prompt"])

        print(f"\nSample {i+1}")
        print(f"Prompt   : {p['prompt'][:120]}...")
        print(f"Baseline : {out_baseline[:200]}")
        print(f"EAGLE-3  : {out_eagle3[:200]}")
        print("-" * 80)

    print("\nManual review required — outputs printed above.")
    print("Look for: fluency, factual consistency, length similarity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url", default="http://localhost:8000")
    parser.add_argument("--eagle3-url", default="http://localhost:8001")
    parser.add_argument("--task", choices=["chat", "code", "summarization"], required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--n-check", type=int, default=10)
    args = parser.parse_args()

    with open(args.prompts_file) as f:
        prompts = [json.loads(l) for l in f]

    if args.task == "summarization":
        check_summarization(args.baseline_url, args.eagle3_url, prompts, args.n_check)
    elif args.task == "code":
        check_code(args.baseline_url, args.eagle3_url, prompts, args.n_check)
    elif args.task == "chat":
        check_chat(args.baseline_url, args.eagle3_url, prompts, args.n_check)


if __name__ == "__main__":
    main()
