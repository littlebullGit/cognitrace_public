# CogniTrace Model Card

## Overview

CogniTrace combines two model roles:

- **Screening score:** an on-device classical ensemble exported to ONNX.
- **Explanation and follow-up:** Gemma 4 E2B, fine-tuned for medical plain-language communication and exported as GGUF Q4_K_M for local llama.cpp/llamadart inference.

Gemma is not used as the Parkinson prediction signal. It receives the ensemble result and app context, then produces cautious plain-language summaries, biomarker education, multilingual explanations, and doctor-prep guidance.

## Artifacts

| Artifact | Location |
| --- | --- |
| ONNX classifier assets | `app/assets/models/` |
| Gemma app integration | `app/lib/services/gemma_service.dart` |
| Model download manager | `app/lib/services/gemma_download_manager.dart` |
| Fine-tuning code | `training/finetune_gemma_qlora.py`, `training/sagemaker/src/train.py` |
| GGUF export code | `training/export_gguf.py` |
| Release gates | `training/ship_gate.py`, `training/results_v3/` |
| Public GGUF URL | `https://huggingface.co/littlebull9/cognitrace-gemma4-medical-GGUF` |

## Intended Use

CogniTrace is a research prototype for adult voice-screening education. It is intended to help a user:

- complete short voice tasks on-device,
- see a screening score and measured biomarkers,
- understand the result in plain language,
- prepare better questions for a clinician,
- keep audio and results local to the phone.

## Out of Scope

CogniTrace does not diagnose Parkinson's disease, prescribe treatment, replace a neurologist, or claim clinical approval. Gemma should not infer disease status from raw audio or override the ONNX score.

## Validation Snapshot

| Check | Result |
| --- | --- |
| Subject-grouped 5-fold CV on 799 Swift-extracted recordings from 59 subjects | 95.74% accuracy, 0.9574 macro-F1, 0.9908 AUC-ROC |
| Mobile parity | Classifier trained on the same 56-feature Swift vectors used by the app |
| Gemma v3 JSON reliability | 30/30 valid responses |
| Gemma v3 adversarial safety | 0 flagged violations across 25 prompts in 5 languages |
| Gemma v3 latency gate | 34.1 ms first-token vs 33.0 ms base, 1.034x |
| Gemma v3 memory gate | +29 MB peak memory vs base in local GGUF eval |
| App-specific practice review | Passed 7 app-shaped prompts |

Detailed result files are in `training/results_v3/`.

## Safety Boundary

The app keeps prediction and explanation separate. The ONNX ensemble produces the score. Gemma turns that score into plain-language, non-diagnostic guidance and refuses treatment or medication advice. The UI and prompts repeatedly frame the result as screening only.

## Data and Privacy

Voice recordings, extracted features, results, and generated explanations stay in app-local storage unless the user chooses to share through iOS. The full fine-tuning JSONL is not committed to this repo; source licenses, curation code, provenance, and eval fixtures are included so the data path is auditable without publishing unnecessary payloads in Git.

## Known Limits

- Validation is internal and based on one open Italian Parkinson voice corpus.
- Real-world microphones, noise, fatigue, illness, language, and clinical population shifts need external validation.
- The bundled reference clips are demonstration samples from the public dataset, not representative clinical evidence.
- Gemma output quality is gated, but not a substitute for clinician-reviewed medical advice.

