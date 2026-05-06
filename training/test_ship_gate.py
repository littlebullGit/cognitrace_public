#!/usr/bin/env python3
"""Regression tests for shipped Gemma gate selection."""

from __future__ import annotations

import json
import sys

import pytest

import ship_gate


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_ship_gate_ignores_optional_broad_comparison_failures(tmp_path, monkeypatch):
    write_json(tmp_path / "gate_json_reliability.json", {"parse_rate": 1.0})
    for lang in ["en", "it", "zh", "es", "fr"]:
        write_json(tmp_path / f"adversarial_{lang}.json", {"violations": 0})
    write_json(
        tmp_path / "gate_latency.json",
        {"candidate_first_token_ms": 34.1, "base_first_token_ms": 33.0},
    )
    write_json(
        tmp_path / "gate_memory.json",
        {"candidate_peak_mb": 3768, "base_peak_mb": 3739},
    )

    # These broad comparison artifacts may be kept for analysis, but they are
    # not release gates for the app-shaped shipped Gemma role.
    write_json(
        tmp_path / "finetune_eval.json",
        {"candidate_fkgl_mean": 8.78, "base_fkgl_mean": 8.33},
    )
    write_json(
        tmp_path / "ab_results.json",
        {"candidate_preferred": 3, "total_rated": 20},
    )

    monkeypatch.setattr(sys, "argv", ["ship_gate.py", "--results-dir", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        ship_gate.main()

    assert excinfo.value.code == 0
