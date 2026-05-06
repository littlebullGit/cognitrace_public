# CogniTrace

**A 60-second on-device voice check for early Parkinson's screening, powered by Gemma 4**

Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) | Health & Sciences Track

## The Problem

Nearly 12 million people worldwide live with Parkinson's disease. Voice and speech changes appear 7 to 11 years before motor symptoms (Fereshtehnejad et al., 2019), but most of the world lacks access to neurologists who can catch these early signs. By the time tremors trigger a clinical visit, over half of dopamine-producing neurons are already gone. CogniTrace puts a screening tool in your pocket.

## What It Does

- Records 3 voice tasks in 60 seconds (sustained vowel, rapid syllables, free speech)
- Extracts 56 acoustic biomarkers on-device (Swift/vDSP/Accelerate)
- Runs stacked ML ensemble (LightGBM + XGBoost + CatBoost via ONNX, <2ms) on all 56 features
- Keeps prediction with the classical ensemble, then uses Gemma 4 E2B for explanation, multilingual education, and doctor-prep on-device
- Voice recording and scoring can run while Gemma downloads. Before the download finishes, the app can score the voice check but cannot generate the AI explanation yet. Once the model is on the phone, AI interpretation, education, and doctor-prep run offline. Audio and results remain on-device, and saved checks can be deleted.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  iPhone 14+ (iOS 16.4+)                              │
│                                                      │
│  Voice → Swift Feature Extraction (56 features)      │
│       → ONNX Ensemble, all 56 features (<2ms)        │
│       → Risk Score                                   │
│       → Gemma 4 E2B Narrative (Metal GPU, ~15s)      │
│                                                      │
│  5 Languages · Fully Offline · Privacy-First         │
└──────────────────────────────────────────────────────┘
```

## Gemma 4 Integration

Gemma 4 E2B runs via llamadart (llama.cpp) with GGUF Q4_K_M quantization. Metal GPU acceleration keeps generation at roughly 15 seconds on an iPhone 14. Memory-tuned params (contextSize=2048, batchSize=512) support both the initial summary and follow-up chat. The initial summary is intentionally risk-based: Gemma receives the final ensemble result and turns it into clear plain-language screening guidance in English, Italian, Chinese, Spanish, and French. It does not reproduce the classifier's internal reasoning or act as a predictive model.

For drill-down, the biomarker table still shows all 56 extracted values, and the intended follow-up chat design is full-data access with cautious guardrails rather than a tiny explainer-only subset.

An offline GPU experiment with Gemma 4 E4B native audio input was near chance on this dataset, so raw-audio Gemma is **not** used as a prediction signal in the app.

## Fine-Tuning Provenance

The first Gemma fine-tuning path was a Kaggle notebook. That run was useful for packaging the dataset and proving the Unsloth workflow, but the Kaggle v7 training run diverged to NaN under fp16. The shipped v3 artifact was produced through the hardened SageMaker path in `training/sagemaker/`, using bf16 on an A10G instance, Gemma chat-template formatting, NaN guards, warmup/cosine scheduling, and local GGUF export.

The app loads `cognitrace-gemma4-medical-v3-Q4_K_M.gguf` from Hugging Face. The Kaggle notebook is kept for reproducibility context; the final model evidence is tied to the SageMaker/local-exported v3 GGUF.

## What We Learned From Trying Gemma Prediction

We explicitly tested whether Gemma could act as a raw-audio prediction model
for Parkinson voice screening. That was the right experiment to run because it
forced the product to answer a hard question directly instead of assuming the
answer from model capability marketing alone.

The result was clear: native-audio Gemma was not good enough for this task. It
was near chance overall, had zero selective sensitivity on the evaluated slice,
and produced a false-reassurance rate that is too high for a health screening
workflow. In other words, the model was not just weaker than the classical
pipeline; it was not responsible to ship as a prediction signal.

That finding improved the design. Rather than forcing Gemma into prediction,
CogniTrace now keeps the roles explicit:

- the ONNX ensemble produces the score
- the biomarker table shows the raw measurements
- Gemma explains the score, teaches what it means, and prepares the user for follow-up

This is a stronger product role than a weak second-opinion classifier, because
it aligns Gemma with what it does well while keeping prediction with the model
that actually performs.

## Demo Flow

The clearest demo story is now:

1. The classical ensemble scores the recording on-device.
2. The raw score and measured 56-marker voice profile remain visible.
3. Gemma turns that same score into a usable screening experience:
   plain-language explanation, multilingual handoff, and doctor discussion prep.
4. Switching languages keeps the score fixed while Gemma rewrites the explanation.
5. The app can say plainly that Gemma was tested for raw-audio prediction and rejected for that role.

## Validation Snapshot

| Check | Result | Limit |
|-------|--------|-------|
| Subject-grouped 5-fold CV on 799 Swift-extracted recordings from 59 subjects | 95.74% accuracy, 0.9574 macro-F1, 0.9908 AUC-ROC | Internal validation on one dataset |
| Leakage control | Recordings from the same subject never appear in both train and validation folds | Not a replacement for external clinical validation |
| Mobile parity | Classifier trained on the same 56-feature Swift vectors used by the app | Real-world microphones and noisy rooms need broader testing |
| Shipped Gemma v3 safety gate | 0 flagged violations across 25 adversarial prompts in 5 languages | Automated checks are not clinical review |
| Shipped Gemma v3 response reliability | 30/30 valid generated responses | Runtime fallback still exists if generation fails |
| Shipped Gemma v3 app-specific review | Passed 7 app-shaped practice prompts covering summaries, moderate follow-up, biomarker explanation, treatment refusal, and Chinese output | LLM-as-judge/manual review, not a clinical study |
| Shipped Gemma v3 runtime gate | First-token latency 34.1 ms vs 33.0 ms base; peak memory +29 MB vs base in local GGUF eval | Local eval does not replace real iPhone UX timing |

The release gate is app-facing: safety, response reliability, runtime, and app-shaped practice review tied to the shipped v3 GGUF in `training/results_v3/`.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Flutter 3.x (Dart) |
| Audio | AVFoundation (16kHz mono PCM) |
| Features | Swift + vDSP/Accelerate (56 biomarkers extracted on-device) |
| Classifier | ONNX Runtime (LGB + XGB + CB ensemble on all 56 extracted features) |
| LLM | Gemma 4 E2B via llamadart (llama.cpp, GGUF Q4_K_M) |
| i18n | 5 languages (EN, IT, ZH, ES, FR) |
| Storage | On-device saved checks in the app container, user-deletable |

## Getting Started

```bash
# Prerequisites: Flutter 3.11+, Xcode 16+, physical iPhone (no simulator)

git clone https://github.com/littlebullGit/cognitrace_public.git
cd cognitrace_public/app
flutter pub get
cd ios && pod install && cd ..
flutter run --release  # Release mode recommended (saves ~400MB debugger overhead)
```

Notes:
- Gemma 4 model downloads automatically on first launch and is required only for AI interpretation, not for recording or scoring
- Requires iOS 16.4+ on physical device (llamadart FFI needs real Metal GPU)
- No API keys needed. The GGUF model is public Apache 2.0.

## Repository Contents

| Path | Purpose |
|------|---------|
| `app/` | Flutter iOS app, Swift audio/features bridge, ONNX models, local knowledge assets, tests |
| `training/` | Fine-tuning/evaluation scripts, license manifest, public-safe fixtures, and v3 result artifacts |
| `.ios-assets/` | Mobile feature parity scripts and reference vectors used during iOS validation |
| `assets/` | Public submission images |
| `docs/` | Public development guide, model card, writeup draft, and privacy policy |
| `SUBMISSION.md` | Kaggle judge-facing checklist and artifact map |

## Dataset

Italian PD Voice & Speech: 65 subjects aged 50+ (PD and age-matched healthy controls), 831 WAV files (IEEE DataPort, open access). Two subjects held out as unseen test samples for bundled demo clips. Final models trained on 59 subjects (799 recordings) using Swift-extracted features for mobile parity.

The full Gemma medical-communication training JSONL is intentionally not committed to GitHub. The repository includes the license manifest, curation scripts, provenance manifest, evaluation prompts, and current result artifacts; see `training/data/README.md` for how to reconstruct or publish the dataset separately.

## Limitations

- This is a screening tool, not a diagnosis. It does not replace a neurologist.
- Trained on a single dataset (Italian PD corpus). Cross-population validation is ongoing.
- Voice recordings were collected under controlled conditions. Real-world noise may affect accuracy.
- Not FDA-approved or CE-marked. Research use only.

## References

1. Fereshtehnejad et al. (2019). Evolution of prodromal Parkinson's disease. *Brain*, 142(7).
2. Favaro et al. (2024). Early PD signs via celebrity speech analysis. *npj Parkinson's Disease*, 10.
3. Cao et al. (2025). Speech biomarkers for PD. *npj Parkinson's Disease*, 11.
4. Little et al. (2009). Dysphonia measurements for PD telemonitoring. *IEEE Trans. Biomed. Eng.*, 56(4).

## License

MIT (app code) | Apache 2.0 (Gemma 4 model)
