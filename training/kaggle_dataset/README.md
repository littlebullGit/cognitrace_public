# CogniTrace Medical Communication Training Data

Instruction pairs for fine-tuning Gemma 4 E2B on medical-to-plain-language simplification.

## Sources (all license-safe)
- PLABA (CC BY 4.0) - biomedical text simplification
- MTS-Dialog (CC BY 4.0) - medical dialogue summarization  
- NHS (OGL v3) - Parkinson's patient information
- MedlinePlus (public domain) - consumer health information

## Format
JSONL with chat messages format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]}
```

## Usage
Create or attach a Kaggle dataset, then reference it in the notebook as:
```
/kaggle/input/cognitrace-medical-communication/medical_communication.jsonl
```

## Before Upload

1. Rebuild or retrieve `training/data/medical_communication.jsonl`.
2. Copy it into this directory before uploading to Kaggle.
3. Replace `INSERT_USERNAME` in `dataset-metadata.json`, or run `training/run_kaggle_training.sh`, which performs that substitution.

The full JSONL is not committed to this GitHub repository; see `training/data/README.md` and `training/LICENSE_MANIFEST.md`.
