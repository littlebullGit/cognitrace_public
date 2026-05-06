# CogniTrace Kaggle Submission Map

This repository is the public code artifact for the Gemma 4 Good Hackathon submission.

## Kaggle Requirements

| Requirement | Where it is covered |
| --- | --- |
| Public code repository | This GitHub repository |
| Documented code | `README.md`, `docs/DEVELOPMENT.md`, app and training READMEs |
| Clear Gemma 4 implementation | `app/lib/services/gemma_service.dart`, `app/lib/services/gemma_download_manager.dart`, `training/` |
| App architecture | `README.md`, `docs/DEVELOPMENT.md` |
| Benchmarks and release evidence | `README.md`, `docs/model-card.md`, `training/results_v3/` |
| Privacy and safety | `docs/privacy-policy.md`, `docs/model-card.md` |
| Writeup support | `docs/kaggle-writeup-final.md` |
| Media gallery | `assets/kaggle-media-gallery/` |

## What Judges Should Inspect

- `app/`: Flutter iOS app with on-device recording, Swift feature extraction, ONNX scoring, Gemma download, Gemma explanation, multilingual UI, and local history.
- `app/assets/models/`: shipped ONNX ensemble and scaler assets used for local scoring.
- `app/assets/reference_audio/`: small public-dataset reference clips used by the bundled sample runner.
- `training/`: Gemma fine-tuning, evaluation, GGUF export, SageMaker path, Kaggle notebook path, and release gates.
- `training/results_v3/`: current shipped-v3 safety, reliability, latency, memory, A/B, and app-practice evidence.
- `docs/model-card.md`: concise model role, artifact map, safety boundary, and limitations.

## Public-Repo Boundary

Included:

- Source code needed to build and inspect the app.
- Small model artifacts required by the app's local ONNX ensemble.
- Public reference WAVs from the Italian PD Voice & Speech dataset.
- Training and evaluation code, public-safe eval prompts, provenance manifests, and v3 result artifacts.
- Kaggle media and public documentation.

Excluded:

- Private git history, local logs, `.omx/`, caches, build outputs, and virtual environments.
- `.env` files, tokens, Apple signing team IDs, App Store metadata, TestFlight notes, and private account state.
- Full fine-tuning JSONL payloads. These are documented through `training/LICENSE_MANIFEST.md`, `training/data/medical_communication_manifest.jsonl`, curation scripts, and Kaggle dataset metadata so they can be reconstructed or hosted as a separate public dataset artifact.
- Large Gemma GGUF/model weights. The app downloads the public GGUF from Hugging Face.

## Repository Standard

This repo follows the pattern used by strong Gemma hackathon submissions: a working app path, a clear model role, a short architecture explanation, public media, reproducible training/eval scripts, concrete benchmark artifacts, safety limits, and privacy boundaries. Prior public examples reviewed while preparing this included health-app and Gemma education repos such as DermaCheck and current Gemma 4 hackathon projects.

