# Training Data

The full medical-communication fine-tuning JSONL is intentionally not committed to this public code repo. It is derived from public, license-reviewed sources, but keeping large generated payloads outside Git keeps the repo focused on code, provenance, and verification.

Included here:

- `eval_prompts.jsonl` - app-facing evaluation prompts.
- `adversarial_prompts.jsonl` - multilingual safety prompts.
- `medical_communication_manifest.jsonl` - provenance preview for the curated instruction pairs.
- `v2_synthetic/generate.py` - generator for app-aligned synthetic examples.

Use `training/LICENSE_MANIFEST.md` for source licenses and attribution requirements. Use `training/curate_medical_data.py`, `training/scrape_nhs_medlineplus.py`, and `training/format_instruction_pairs.py` to rebuild the local JSONL before running training.

Runtime paths expected by training scripts:

```text
training/data/medical_communication.jsonl
training/data/medical_communication_val.jsonl
```

Those files should be produced locally or attached through a Kaggle Dataset artifact, not committed directly to GitHub.

