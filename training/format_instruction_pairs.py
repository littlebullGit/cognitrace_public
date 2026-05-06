#!/usr/bin/env python3
"""Merge all source JSONL files into a single training-ready dataset.

Reads:
  training/data/*_pairs.jsonl

Writes:
  training/data/medical_communication.jsonl      -- 90% train split (feeds Unsloth SFTTrainer)
  training/data/medical_communication_val.jsonl  -- 10% validation split

Output format (Gemma chat template):
  {"messages": [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]}

Steps performed:
  1. Load all *_pairs.jsonl source files
  2. Validate schema: each entry must have messages list with user + model roles
  3. Validate lengths: instruction 10-500 words, response 20-500 words
  4. Deduplicate on exact instruction text
  5. Add _source field for audit (stripped before writing final files)
  6. Shuffle with seed 42
  7. 90/10 train/val split
  8. Print per-source statistics
"""

import argparse
import json
import random
from pathlib import Path


SEED = 42
TRAIN_FRACTION = 0.90

MIN_INSTRUCTION_WORDS = 10
MAX_INSTRUCTION_WORDS = 500
MIN_RESPONSE_WORDS = 20
MAX_RESPONSE_WORDS = 500

BANNED_RESPONSE_PHRASES = [
    "you have", "diagnosed with", "treatment plan",
    "prescribe", "you should take", "you need to take",
    "your diagnosis",
]


def load_source(path: Path) -> list[dict]:
    pairs = []
    source_name = path.stem
    bad_schema = 0
    bad_length = 0
    bad_safety = 0

    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue

            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  WARN: {path.name}:{lineno}: invalid JSON ({exc}), skipping")
                continue

            messages = entry.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                bad_schema += 1
                continue

            user_msg = next(
                (m for m in messages if m.get("role") == "user"), None
            )
            model_msg = next(
                (m for m in messages if m.get("role") == "model"), None
            )

            if user_msg is None or model_msg is None:
                bad_schema += 1
                continue

            instruction = user_msg.get("content", "").strip()
            response = model_msg.get("content", "").strip()

            if not instruction or not response:
                bad_schema += 1
                continue

            instruction_words = len(instruction.split())
            response_words = len(response.split())

            if not (MIN_INSTRUCTION_WORDS <= instruction_words <= MAX_INSTRUCTION_WORDS):
                bad_length += 1
                continue

            if not (MIN_RESPONSE_WORDS <= response_words <= MAX_RESPONSE_WORDS):
                bad_length += 1
                continue

            response_lower = response.lower()
            if any(phrase in response_lower for phrase in BANNED_RESPONSE_PHRASES):
                bad_safety += 1
                continue

            pairs.append({
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "model", "content": response},
                ],
                "_source": source_name,
                "_source_id": entry.get("_source_id", ""),
                "_source_url": entry.get("_source_url", ""),
            })

    if bad_schema:
        print(f"  {path.name}: {bad_schema} entries dropped (schema invalid)")
    if bad_length:
        print(f"  {path.name}: {bad_length} entries dropped (length out of range)")
    if bad_safety:
        print(f"  {path.name}: {bad_safety} entries dropped (banned phrases in response)")

    return pairs


def deduplicate(pairs: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    unique: list[dict] = []
    for entry in pairs:
        key = entry["messages"][0]["content"]
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique, len(pairs) - len(unique)


def strip_source_field(pairs: list[dict]) -> list[dict]:
    return [
        {"messages": entry["messages"]}
        for entry in pairs
    ]


def write_jsonl(pairs: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")


def print_stats(pairs: list[dict], label: str) -> None:
    from collections import Counter

    source_counts: Counter = Counter(p.get("_source", "unknown") for p in pairs)
    instruction_lengths = [
        len(p["messages"][0]["content"].split()) for p in pairs
    ]
    response_lengths = [
        len(p["messages"][1]["content"].split()) for p in pairs
    ]

    avg_instr = sum(instruction_lengths) / max(len(instruction_lengths), 1)
    avg_resp = sum(response_lengths) / max(len(response_lengths), 1)

    print(f"\n{label}: {len(pairs)} pairs total")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")
    print(f"  Avg instruction length: {avg_instr:.1f} words")
    print(f"  Avg response length:    {avg_resp:.1f} words")


def data_dir() -> Path:
    d = Path(__file__).parent / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all *_pairs.jsonl sources into the final training dataset."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing *_pairs.jsonl files (default: training/data/)"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Shuffle seed (default: {SEED})"
    )
    parser.add_argument(
        "--train-fraction", type=float, default=TRAIN_FRACTION,
        help=f"Fraction of data to use for train split (default: {TRAIN_FRACTION})"
    )
    args = parser.parse_args()

    dest = args.data_dir or data_dir()
    source_files = sorted(dest.glob("*_pairs.jsonl"))

    if not source_files:
        print(f"No *_pairs.jsonl files found in {dest}")
        print("Run curate_medical_data.py and scrape_nhs_medlineplus.py first.")
        raise SystemExit(1)

    print(f"Found {len(source_files)} source files:")
    for f in source_files:
        print(f"  {f.name}")

    all_pairs: list[dict] = []
    for src in source_files:
        print(f"\nLoading {src.name}...")
        loaded = load_source(src)
        print(f"  Loaded {len(loaded)} valid pairs")
        all_pairs.extend(loaded)

    print_stats(all_pairs, "Before dedup")

    all_pairs, n_dupes = deduplicate(all_pairs)
    print(f"\nDeduplication: removed {n_dupes} exact-match duplicates")

    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)

    cutoff = int(len(all_pairs) * args.train_fraction)
    train_pairs = all_pairs[:cutoff]
    val_pairs = all_pairs[cutoff:]

    print_stats(all_pairs, "After dedup (combined)")

    # Write audit sidecar with per-pair provenance before stripping source fields
    audit_path = dest / "medical_communication_manifest.jsonl"
    with audit_path.open("w", encoding="utf-8") as fh:
        for i, p in enumerate(all_pairs):
            audit_entry = {
                "pair_id": i,
                "source": p.get("_source", "unknown"),
                "source_id": p.get("_source_id", ""),
                "source_url": p.get("_source_url", ""),
                "instruction_preview": p["messages"][0]["content"][:80],
            }
            fh.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
    print(f"\nAudit sidecar: {audit_path} ({len(all_pairs)} entries)")

    train_clean = strip_source_field(train_pairs)
    val_clean = strip_source_field(val_pairs)

    train_path = dest / "medical_communication.jsonl"
    val_path = dest / "medical_communication_val.jsonl"

    write_jsonl(train_clean, train_path)
    write_jsonl(val_clean, val_path)

    print(f"\nOutput:")
    print(f"  Train: {len(train_clean)} pairs -> {train_path}")
    print(f"  Val:   {len(val_clean)} pairs  -> {val_path}")
    print(f"  Total: {len(all_pairs)} pairs")

    if len(all_pairs) < 550:
        print(
            f"\nWARN: only {len(all_pairs)} pairs generated. "
            "Target is 550-800. Check that all source scripts ran successfully."
        )
    else:
        print(f"\nTarget met: {len(all_pairs)} pairs (target: 550-800).")


if __name__ == "__main__":
    main()
