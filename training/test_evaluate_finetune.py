#!/usr/bin/env python3
"""Regression tests for local Gemma fine-tune evaluation helpers."""

from __future__ import annotations

import json
import gc

import evaluate_finetune


def test_run_evaluation_does_not_keep_base_and_candidate_loaded_together(tmp_path, monkeypatch):
    prompts_path = tmp_path / "eval_prompts.jsonl"
    prompts_path.write_text(
        "\n".join([
            json.dumps({"prompt": "Explain a low result.", "category": "screening"}),
            json.dumps({"prompt": "Explain an elevated result.", "category": "screening"}),
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate_finetune, "EVAL_PROMPTS_PATH", str(prompts_path))
    monkeypatch.setattr(evaluate_finetune, "compute_fkgl", lambda text: 6.0)

    state = {"active": 0, "max_active": 0}

    class FakeLlama:
        def __init__(self, label: str):
            self.label = label
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])

        def close(self):
            if state["active"] > 0:
                state["active"] -= 1

        def __del__(self):
            self.close()

    def fake_load_llama(path: str):
        return FakeLlama(path)

    def fake_generate_response(llm: FakeLlama, prompt: str):
        return f"{llm.label}: {prompt}"

    monkeypatch.setattr(evaluate_finetune, "load_llama", fake_load_llama)
    monkeypatch.setattr(evaluate_finetune, "generate_response", fake_generate_response)

    evaluate_finetune.run_evaluation("base.gguf", "candidate.gguf", str(tmp_path))
    gc.collect()

    assert state["max_active"] == 1
