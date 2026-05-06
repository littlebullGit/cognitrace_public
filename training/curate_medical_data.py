#!/usr/bin/env python3
"""Download, filter, and format PLABA + MTS-Dialog datasets.

Outputs:
  training/data/plaba_pairs.jsonl      -- biomedical-to-plain translation pairs
  training/data/mts_dialog_pairs.jsonl -- doctor-patient dialogue pairs

Both files use Gemma chat format:
  {"messages": [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]}

License:
  PLABA     -- CC BY 4.0 (cite: Luo et al., 2022)
  MTS-Dialog -- CC BY 4.0 (cite: Ben Abacha et al., 2023)
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEALTH_TERMS = {
    "voice", "speech", "neurodegenerative", "parkinson", "tremor",
    "movement", "cognitive", "screening", "biomarker", "diagnostic",
    "clinical", "patient", "symptom", "motor", "muscle", "brain",
    "neurolog", "dysphonia", "vocal", "articulation", "dementia",
    "alzheimer", "gait", "rigidity", "bradykinesia", "dopamine",
    "disorder", "disease", "treatment", "therapy", "medication",
    "physician", "doctor", "diagnosis", "prognosis", "assessment",
    "evaluation", "test", "score", "risk", "health", "medical",
    "hear", "listen", "commun", "language", "memory", "aging",
}

# MTS-Dialog raw CSV URLs (CC BY 4.0, abachaa/MTS-Dialog on GitHub)
MTS_DIALOG_URLS = {
    "train": (
        "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/"
        "Main-Dataset/MTS-Dialog-TrainingSet.csv"
    ),
    "test1": (
        "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/"
        "Main-Dataset/MTS-Dialog-TestSet1-InputOnly.csv"
    ),
    "test2": (
        "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/"
        "Main-Dataset/MTS-Dialog-TestSet2-InputOnly.csv"
    ),
}

# Instruction templates for PLABA pairs
PLABA_INSTRUCTIONS = [
    "Translate this biomedical text to plain language a patient can understand: {text}",
    "Explain this clinical finding at a 6th-grade reading level: {text}",
    "Rewrite this for a patient with no medical background: {text}",
]

# Instruction templates for MTS-Dialog pairs
MTS_INSTRUCTIONS = [
    "Based on this doctor-patient exchange, write a clear patient-friendly summary: {text}",
    "Summarise this clinical conversation in simple language a patient can understand: {text}",
]

MTS_PREP_INSTRUCTIONS = [
    "Help a patient prepare to discuss these symptoms with their doctor: {text}",
    "What should a patient write down before a doctor visit about these symptoms? {text}",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def out_dir() -> Path:
    here = Path(__file__).parent
    d = here / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def contains_health_term(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in HEALTH_TERMS)


def word_count(text: str) -> int:
    return len(text.split())


def make_pair(instruction: str, response: str, source: str = "", source_id: str = "", source_url: str = "") -> dict:
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "model", "content": response},
        ],
        "_source": source,
        "_source_id": source_id,
        "_source_url": source_url,
    }


def write_jsonl(pairs: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(pairs)} pairs to {path}")


# ---------------------------------------------------------------------------
# PLABA
# ---------------------------------------------------------------------------

def curate_plaba(seed: int) -> list[dict]:
    """Download PLABA from GitHub/OSF and produce filtered instruction-response pairs."""
    import requests as req

    plaba_url = "https://osf.io/download/4kp7v/"
    print(f"Downloading PLABA data.json from OSF...")

    resp = req.get(plaba_url, timeout=120)
    resp.raise_for_status()

    raw = resp.json()
    pairs: list[dict] = []
    skipped = 0

    # PLABA JSON: question_id -> abstract_id -> {Title, abstract: {sent_id: text}, adaptations: {adapt_id: {sent_id: text}}}
    items = []
    if isinstance(raw, dict):
        for question_id, abstracts in raw.items():
            if not isinstance(abstracts, dict):
                continue
            for abstract_id, entry in abstracts.items():
                if not isinstance(entry, dict):
                    continue
                source_sents = entry.get("abstract", {})
                adaptations = entry.get("adaptations", {})
                if not isinstance(source_sents, dict) or not isinstance(adaptations, dict):
                    continue
                for adapt_id, adapted_sents in adaptations.items():
                    if not isinstance(adapted_sents, dict):
                        continue
                    for sent_id, src_text in source_sents.items():
                        tgt_text = adapted_sents.get(sent_id, "")
                        if isinstance(src_text, str) and isinstance(tgt_text, str) and src_text.strip() and tgt_text.strip():
                            items.append({"biomedical": src_text.strip(), "plain": tgt_text.strip()})

    print(f"  Parsed {len(items)} sentence pairs from PLABA")

    for item in items:
        biomedical = item["biomedical"]
        plain = item["plain"]

        if word_count(biomedical) < 8 or word_count(biomedical) > 250:
            skipped += 1
            continue
        if word_count(plain) < 8 or word_count(plain) > 250:
            skipped += 1
            continue

        if not contains_health_term(biomedical) and not contains_health_term(plain):
            skipped += 1
            continue

        template = PLABA_INSTRUCTIONS[len(pairs) % len(PLABA_INSTRUCTIONS)]
        instruction = template.format(text=biomedical)
        pairs.append(make_pair(
            instruction, plain,
            source="PLABA", source_id=f"plaba_{len(pairs)}",
            source_url="https://github.com/attal-kush/PLABA",
        ))

    print(f"  PLABA: {len(pairs)} pairs kept, {skipped} skipped")
    return pairs


# ---------------------------------------------------------------------------
# MTS-Dialog
# ---------------------------------------------------------------------------

def _fetch_csv_lines(url: str) -> list[str]:
    """Fetch a URL and return lines as strings, with simple retry logic."""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read().decode("utf-8").splitlines()
        except Exception as exc:
            if attempt == 2:
                raise
            print(f"  Retry {attempt + 1}/3 for {url} ({exc})")
            time.sleep(2 ** attempt)
    return []


def _parse_csv_rows(lines: list[str]) -> list[dict]:
    """
    Minimal CSV parser that handles quoted fields with embedded commas and newlines.
    Returns list of dicts keyed by header row.
    """
    import csv
    import io
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    return list(reader)


def _extract_symptom_phrases(dialogue: str) -> str:
    """
    Pull out short phrases that describe patient symptoms from a dialogue.
    Heuristic: look for lines where patient describes feelings/symptoms.
    Returns a compact summary string, not structured data.
    """
    symptom_keywords = (
        "feel", "pain", "hurt", "ache", "dizzy", "tired", "weak", "numb",
        "swell", "bleed", "cough", "breath", "nausea", "vomit", "fever",
        "headache", "symptom", "trouble", "difficult", "problem", "worse",
        "started", "began", "noticed", "since",
    )
    phrases = []
    for line in dialogue.splitlines():
        lower = line.lower()
        if any(kw in lower for kw in symptom_keywords):
            # Remove speaker label if present ("Patient:", "Doctor:", etc.)
            cleaned = re.sub(r"^(patient|doctor|physician|clinician)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
            if cleaned and len(cleaned) > 10:
                phrases.append(cleaned)

    if not phrases:
        return dialogue[:300]  # Fall back to truncated raw text
    return " ".join(phrases[:5])


def curate_mts_dialog(seed: int) -> list[dict]:
    """Download MTS-Dialog CSVs from GitHub and produce instruction-response pairs."""
    pairs: list[dict] = []
    skipped = 0

    for split_name, url in MTS_DIALOG_URLS.items():
        print(f"  Fetching MTS-Dialog '{split_name}' from GitHub...")
        try:
            lines = _fetch_csv_lines(url)
        except Exception as exc:
            print(f"  WARNING: could not fetch {split_name} ({exc}), skipping")
            continue

        rows = _parse_csv_rows(lines)
        print(f"  Parsed {len(rows)} rows from '{split_name}'")

        for row in rows:
            # Column names used in MTS-Dialog: dialogue, section_header, section_text
            dialogue = (
                row.get("dialogue")
                or row.get("Dialogue")
                or row.get("DIALOGUE")
                or ""
            ).strip()
            section_header = (
                row.get("section_header")
                or row.get("Section_Header")
                or row.get("SECTION_HEADER")
                or ""
            ).strip()
            section_text = (
                row.get("section_text")
                or row.get("Section_Text")
                or row.get("SECTION_TEXT")
                or ""
            ).strip()

            if not dialogue:
                skipped += 1
                continue

            # Length guard
            if word_count(dialogue) < 20 or word_count(dialogue) > 600:
                skipped += 1
                continue

            # Dialogue -> patient-friendly summary pair
            if section_text and word_count(section_text) >= 15:
                template_idx = len(pairs) % len(MTS_INSTRUCTIONS)
                template = MTS_INSTRUCTIONS[template_idx]
                instruction = template.format(text=dialogue[:800])
                # The response is the clinical note section reworded for patients.
                # We keep the original clinical note as the response base since
                # MTS-Dialog note sections are already relatively concise.
                response = section_text
                if section_header:
                    response = f"{section_header}: {section_text}"
                pairs.append(make_pair(
                    instruction, response,
                    source="MTS-Dialog", source_id=f"mts_{len(pairs)}",
                    source_url="https://github.com/abachaa/MTS-Dialog",
                ))

            # Symptom prep pair (uses heuristic extraction, no LLM)
            symptom_summary = _extract_symptom_phrases(dialogue)
            if symptom_summary and len(symptom_summary) > 20:
                template_idx = len(pairs) % len(MTS_PREP_INSTRUCTIONS)
                template = MTS_PREP_INSTRUCTIONS[template_idx]
                instruction = template.format(text=symptom_summary)
                # Response: encourage the patient to note when symptoms started,
                # how often they occur, and what makes them better or worse.
                response = (
                    "Before your appointment, write down: when these symptoms first started, "
                    "how often they occur, what makes them better or worse, any other symptoms "
                    "noticed recently, and questions to ask your doctor. "
                    "Bring any recent test results or medication lists."
                )
                pairs.append(make_pair(
                    instruction, response,
                    source="MTS-Dialog", source_id=f"mts_prep_{len(pairs)}",
                    source_url="https://github.com/abachaa/MTS-Dialog",
                ))

    print(f"  MTS-Dialog: {len(pairs)} pairs produced, {skipped} rows skipped")
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and format PLABA + MTS-Dialog for Gemma fine-tuning."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for template cycling (default: 42)"
    )
    parser.add_argument(
        "--skip-plaba", action="store_true",
        help="Skip PLABA download (useful if already cached)"
    )
    parser.add_argument(
        "--skip-mts", action="store_true",
        help="Skip MTS-Dialog download"
    )
    args = parser.parse_args()

    dest = out_dir()

    if not args.skip_plaba:
        print("\n=== PLABA ===")
        plaba_pairs = curate_plaba(seed=args.seed)
        write_jsonl(plaba_pairs, dest / "plaba_pairs.jsonl")
    else:
        print("Skipping PLABA (--skip-plaba set)")

    if not args.skip_mts:
        print("\n=== MTS-Dialog ===")
        mts_pairs = curate_mts_dialog(seed=args.seed)
        write_jsonl(mts_pairs, dest / "mts_dialog_pairs.jsonl")
    else:
        print("Skipping MTS-Dialog (--skip-mts set)")

    print("\nDone. Check training/data/ for output files.")


if __name__ == "__main__":
    main()
