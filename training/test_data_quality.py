#!/usr/bin/env python3
"""Validate curated training data before fine-tuning.

Run with: pytest training/test_data_quality.py -v
"""

import json
import os

import pytest
import textstat

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "medical_communication.jsonl")

BANNED_PHRASES = [
    "you have",
    "diagnosed with",
    "treatment plan",
    "prescribe",
    "you should take",
    "you need to take",
    "your diagnosis",
]

MIN_INSTRUCTION_WORDS = 10
MAX_INSTRUCTION_WORDS = 500
MIN_RESPONSE_WORDS = 20
MAX_RESPONSE_WORDS = 500
MAX_FKGL_GRADE = 12.0
MIN_FKGL_COMPLIANCE_RATE = 0.50
MIN_DATASET_SIZE = 500


def load_dataset() -> list[dict]:
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Dataset not found: {DATA_PATH}. Run curation scripts first.")
    entries = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_minimum_dataset_size(dataset):
    assert len(dataset) >= MIN_DATASET_SIZE, (
        f"Dataset has {len(dataset)} pairs, need at least {MIN_DATASET_SIZE}. "
        "Run curation scripts and check source downloads."
    )


def test_jsonl_format(dataset):
    for i, entry in enumerate(dataset):
        assert "messages" in entry, f"Entry {i} missing 'messages' field"
        messages = entry["messages"]
        assert isinstance(messages, list), f"Entry {i}: 'messages' must be a list"
        assert len(messages) >= 2, f"Entry {i}: expected at least 2 messages (user + model)"

        roles = [m.get("role") for m in messages]
        assert "user" in roles, f"Entry {i}: no 'user' role found in {roles}"
        assert "model" in roles, f"Entry {i}: no 'model' role found in {roles}"

        for j, msg in enumerate(messages):
            assert "role" in msg, f"Entry {i}, message {j}: missing 'role'"
            assert "content" in msg, f"Entry {i}, message {j}: missing 'content'"


def test_instruction_length(dataset):
    violations = []
    for i, entry in enumerate(dataset):
        messages = entry.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            words = len(content.split())
            role = msg.get("role")
            if role == "user" and not (MIN_INSTRUCTION_WORDS <= words <= MAX_INSTRUCTION_WORDS):
                violations.append(f"Entry {i} user message: {words} words (expected {MIN_INSTRUCTION_WORDS}-{MAX_INSTRUCTION_WORDS})")
            elif role == "model" and not (MIN_RESPONSE_WORDS <= words <= MAX_RESPONSE_WORDS):
                violations.append(f"Entry {i} model message: {words} words (expected {MIN_RESPONSE_WORDS}-{MAX_RESPONSE_WORDS})")

    assert not violations, f"{len(violations)} length violations:\n" + "\n".join(violations[:10])


def test_readability(dataset):
    """Report FKGL distribution. This is informational for training data.
    Model-output readability can be inspected with evaluate_finetune.py."""
    grades = []

    for entry in dataset:
        for msg in entry.get("messages", []):
            if msg.get("role") == "model":
                content = msg.get("content", "")
                if len(content.split()) >= 20:
                    grades.append(textstat.flesch_kincaid_grade(content))

    if not grades:
        pytest.skip("No model responses long enough to score.")

    avg_grade = sum(grades) / len(grades)
    below_12 = sum(1 for g in grades if g <= 12.0) / len(grades)
    below_8 = sum(1 for g in grades if g <= 8.0) / len(grades)

    print(f"\n  Training data readability: avg FKGL={avg_grade:.1f}, "
          f"{below_12:.0%} below grade 12, {below_8:.0%} below grade 8")

    assert avg_grade < 16.0, (
        f"Average FKGL {avg_grade:.1f} is unreasonably high. "
        "Training data may be corrupted or misformatted."
    )


def test_no_diagnostic_claims(dataset):
    violations = []
    for i, entry in enumerate(dataset):
        for msg in entry.get("messages", []):
            if msg.get("role") == "model":
                content = msg.get("content", "").lower()
                found = [phrase for phrase in BANNED_PHRASES if phrase in content]
                if found:
                    violations.append(f"Entry {i}: found {found!r} in model response")

    assert not violations, (
        f"{len(violations)} entries contain diagnostic/prescriptive language:\n"
        + "\n".join(violations[:10])
    )


def test_no_empty_fields(dataset):
    violations = []
    for i, entry in enumerate(dataset):
        for j, msg in enumerate(entry.get("messages", [])):
            content = msg.get("content", "")
            if not content or not content.strip():
                violations.append(f"Entry {i}, message {j} (role={msg.get('role')}): empty content")

    assert not violations, (
        f"{len(violations)} empty content fields:\n" + "\n".join(violations[:10])
    )


MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "data", "medical_communication_manifest.jsonl")
VAL_PATH = os.path.join(os.path.dirname(__file__), "data", "medical_communication_val.jsonl")


def test_manifest_exists_and_matches_total():
    assert os.path.exists(MANIFEST_PATH), "Audit manifest missing"
    assert os.path.exists(VAL_PATH), "Validation split missing"

    with open(DATA_PATH) as f:
        train_count = sum(1 for line in f if line.strip())
    with open(VAL_PATH) as f:
        val_count = sum(1 for line in f if line.strip())
    with open(MANIFEST_PATH) as f:
        manifest_count = sum(1 for line in f if line.strip())

    assert manifest_count == train_count + val_count, (
        f"Manifest rows ({manifest_count}) != train ({train_count}) + val ({val_count})"
    )


def test_manifest_provenance_not_blank():
    with open(MANIFEST_PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    blank_source = sum(1 for r in rows if not r.get("source"))
    blank_id = sum(1 for r in rows if not r.get("source_id"))
    blank_url = sum(1 for r in rows if not r.get("source_url"))

    assert blank_source == 0, f"{blank_source}/{len(rows)} rows have blank source"
    assert blank_id == 0, f"{blank_id}/{len(rows)} rows have blank source_id"
    assert blank_url == 0, f"{blank_url}/{len(rows)} rows have blank source_url"
