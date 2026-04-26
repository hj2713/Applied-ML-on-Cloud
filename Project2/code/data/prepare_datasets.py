"""
Prepare benchmark prompts from three datasets.

Downloads from HuggingFace and samples N prompts per task type.
Output: data/prompts/{chat,code,summarization}_N.jsonl

Run:
    python data/prepare_datasets.py --n-samples 50
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Instruction prefix for summarization — tells the model what to do
SUMMARIZATION_PREFIX = "Summarize the following article in 3-4 sentences:\n\n"

# Max input chars to keep prompts at reasonable token length (~512 tokens)
MAX_ARTICLE_CHARS = 2000


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(records)} prompts → {path}")


def prepare_chat(n: int, seed: int) -> list[dict]:
    """Sample n chat prompts from ShareGPT."""
    print("Loading ShareGPT (chat)...")
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
    )
    random.seed(seed)

    records = []
    candidates = list(ds)
    random.shuffle(candidates)

    for item in candidates:
        convs = item.get("conversations", [])
        if not convs:
            continue
        # Take first human turn as the prompt
        first_human = next(
            (c["value"] for c in convs if c.get("from") == "human"), None
        )
        if not first_human or len(first_human.strip()) < 20:
            continue
        records.append(
            {
                "id": f"chat_{len(records):04d}",
                "task": "chat",
                "prompt": first_human.strip()[:1500],  # cap at ~375 tokens
            }
        )
        if len(records) == n:
            break

    return records


def prepare_code(n: int, seed: int) -> list[dict]:
    """Sample n code prompts from HumanEval."""
    print("Loading HumanEval (code)...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    items = list(ds)

    random.seed(seed)
    random.shuffle(items)

    records = []
    for item in items[:n]:
        records.append(
            {
                "id": f"code_{len(records):04d}",
                "task": "code",
                "prompt": item["prompt"].strip(),
                # Store canonical solution for Pass@1 quality check later
                "canonical_solution": item["canonical_solution"],
                "entry_point": item["entry_point"],
            }
        )

    # HumanEval only has 164 problems — repeat if n > 164
    if n > len(items):
        print(f"  Warning: HumanEval has {len(items)} problems, requested {n}. Repeating.")
        while len(records) < n:
            extra = {**records[len(records) % len(items)]}
            extra["id"] = f"code_{len(records):04d}"
            records.append(extra)

    return records[:n]


def prepare_summarization(n: int, seed: int) -> list[dict]:
    """Sample n summarization prompts from CNN/DailyMail."""
    print("Loading CNN/DailyMail (summarization)...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
    items = list(ds)

    random.seed(seed)
    random.shuffle(items)

    records = []
    for item in items:
        article = item["article"].strip()
        if len(article) < 200:
            continue
        # Truncate long articles — we want ~512 input tokens, not 2k
        truncated = article[:MAX_ARTICLE_CHARS]
        records.append(
            {
                "id": f"summarization_{len(records):04d}",
                "task": "summarization",
                "prompt": SUMMARIZATION_PREFIX + truncated,
                # Store reference summary for ROUGE scoring later
                "reference_summary": item["highlights"],
            }
        )
        if len(records) == n:
            break

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50, help="Prompts per task type")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n = args.n_samples
    seed = args.seed

    print(f"\nPreparing {n} prompts per task (seed={seed})\n")

    chat_records = prepare_chat(n, seed)
    _save_jsonl(chat_records, PROMPTS_DIR / f"chat_{n}.jsonl")

    code_records = prepare_code(n, seed)
    _save_jsonl(code_records, PROMPTS_DIR / f"code_{n}.jsonl")

    summ_records = prepare_summarization(n, seed)
    _save_jsonl(summ_records, PROMPTS_DIR / f"summarization_{n}.jsonl")

    print(f"\nDone. Files written to {PROMPTS_DIR}/")
    print("  Task breakdown:")
    print(f"    chat           : {len(chat_records)} prompts")
    print(f"    code           : {len(code_records)} prompts")
    print(f"    summarization  : {len(summ_records)} prompts")


if __name__ == "__main__":
    main()
