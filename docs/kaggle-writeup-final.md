# CogniTrace: 60-Second On-Device Parkinson's Voice Screening with Gemma 4

Subtitle: A privacy-first iPhone prototype that turns voice biomarkers into grounded, multilingual follow-up guidance.

Track: Health & Sciences

## Motivation

Parkinson's disease is often recognized only after visible motor symptoms appear, but voice and speech changes can appear 7 to 11 years before the motor symptoms that usually trigger clinical suspicion, including hand tremor. That creates a practical question: can a phone help people notice a meaningful signal earlier, privately, and in language they can act on?

CogniTrace was built around that gap. It is not a diagnostic product. It is a local-first screening prototype for the moment before a specialist visit, when a person or family member needs a private check, a clear explanation, and concrete questions to bring to a clinician.

## What CogniTrace Does

CogniTrace is a working iPhone prototype for adults 50 and older. The user completes three short voice tasks: a sustained vowel, rapid syllables, and free speech. The app extracts 56 acoustic biomarkers on device using Swift, vDSP, and Accelerate, then runs an ONNX Runtime stacked ensemble of LightGBM, XGBoost, and CatBoost. That deterministic model returns a low, moderate, or elevated screening score in under 2 ms.

Gemma 4 has a separate role. The ensemble scores the voice check. Gemma explains the result, translates it, teaches the relevant biomarkers, and prepares the user for doctor follow-up. The prompt is grounded in the current score, risk band, and biomarker context, so the assistant stays tied to the user's actual screening state rather than giving generic medical advice.

The app is local-first. Recording and ONNX scoring can run while the Gemma model downloads. Before the download finishes, CogniTrace can score the voice check but cannot generate the AI explanation. Once the roughly 2.7 GB GGUF file is on the phone, all Gemma interpretation runs offline. Audio history and prior checks can be saved locally, and the user can delete saved checks. Audio stays on the device.

## Prior Work and the Gap

Voice-based Parkinson screening is an active research area. Sage Bionetworks' mPower study showed that phones can collect Parkinson-related voice and movement signals at scale through Apple ResearchKit. Academic groups have studied speech changes over time, and commercial voice-biomarker platforms exist.

CogniTrace targets a narrower product gap: a consumer-facing, on-device workflow that combines voice screening with an on-device LLM explanation layer. Many practical systems are cloud-based, B2B, research-only, or clinician-facing. CogniTrace is built for the moment before a specialist visit, when a person or family member needs a private check, a clear explanation, and concrete questions to bring to a clinician.

## Why This Uses Gemma 4

In a health screening app, the dangerous moment is the handoff from a number to a human decision. A score can frighten people, confuse people, or get ignored. CogniTrace uses Gemma 4 E2B as the interpretation layer between acoustic biomarkers and the next step.

Gemma 4 provides:

- Plain-language summaries for low, moderate, and elevated results.
- Multilingual explanations in English, Italian, Chinese, Spanish, and French.
- Biomarker education for features such as jitter, shimmer, pitch stability, and syllable timing.
- Doctor-visit preparation, including questions to ask and what information to bring.
- Guardrails that keep the assistant inside screening education and follow-up preparation.

I also tested a more ambitious path: using Gemma as a raw-audio predictor. That experiment failed the product standard. Native-audio Gemma was near chance on this dataset and created too much false reassurance risk. The final architecture keeps prediction in the validated classifier and uses Gemma where it adds safety, clarity, language access, and trust.

## Architecture

The shipped flow is:

```text
iPhone microphone
  -> AVFoundation 16 kHz mono recording
  -> Swift feature extraction, 56 acoustic biomarkers
  -> ONNX Runtime stacked ensemble, all 56 features
  -> Screening result and biomarker table
  -> Gemma 4 E2B GGUF through llamadart and llama.cpp
  -> Plain-language summary, translation, education, and follow-up prep
```

The classifier is trained on the same Swift-extracted feature vectors used by the app, so mobile inference matches the training feature contract. The Gemma runtime uses GGUF Q4_K_M quantization through llama.cpp with Metal acceleration. On an iPhone 14, the classifier score arrives almost immediately, while the first Gemma summary takes about 15 seconds.

This split is intentional. Health screening needs accountable roles. CogniTrace makes the measured signal visible, keeps the deterministic score path separate from generated text, and gives Gemma a constrained prompt containing only the current score, risk band, relevant biomarkers, and safety instructions.

## Training Path and Proof Artifact

Fine-tuning was necessary because the zero-shot base model struggled to reliably guarantee the rigid, app-shaped JSON structure required by Swift, and occasionally deviated into generic medical advice. Fine-tuning forced Gemma to adhere strictly to safety guardrails and stay grounded solely in the user's acoustic screening state.

The shipped model came from a hardened SageMaker and Unsloth path. After early Kaggle training runs diverged to NaN under fp16, moving to SageMaker let me train with bf16 on an A10G instance. I preserved Gemma chat-template formatting, froze unused multimodal towers for the text-only task, added NaN guards, used warmup and cosine scheduling, and exported the exact local GGUF loaded by the app: `cognitrace-gemma4-medical-v3-Q4_K_M.gguf`.

I reran evidence against that exact v3 artifact. The app-facing gates passed:

- 0/25 adversarial safety violations across English, Italian, Chinese, Spanish, and French.
- 30/30 valid generated responses in the reliability check.
- First-token latency within 1.034x of the base GGUF.
- Peak memory only +29 MB over base in local GGUF evaluation.
- App-shaped manual review covering initial summaries, moderate follow-up, biomarker explanation, treatment refusal, and Chinese output.

## Classifier Validation

The voice classifier was evaluated with subject-grouped 5-fold cross-validation on 799 Swift-extracted recordings from 59 subjects. Recordings from the same subject never appear in both train and validation folds. The current internal result is 95.74% accuracy, 0.9574 macro-F1, and 0.9908 AUC-ROC.

Those numbers are promising, but the scope is limited. CogniTrace is a screening prototype and cannot diagnose Parkinson's disease. It was trained on one open Italian Parkinson voice corpus. Real-world deployment would require external validation across languages, phones, microphones, rooms, fatigue states, respiratory illness, and clinical populations.

## Impact

CogniTrace is designed for the gap between "I wonder if something is changing" and "I can get a specialist appointment." A private voice check on a phone cannot replace a neurologist. It can help people notice patterns, save prior checks, delete their data, and walk into a clinical conversation with clearer questions.

That timing matters. CogniTrace cannot prevent Parkinson's disease, but screening can make clinician-guided early intervention possible. When a voice check flags a concerning pattern, the user can seek care sooner, while evidence-backed steps are still actionable: exercise and physical therapy habits, symptom treatment when appropriate, and longitudinal monitoring. High-intensity treadmill exercise has been shown safe in newly diagnosed, untreated Parkinson's disease and associated with less motor-score worsening over six months, while medication evidence supports symptom benefit rather than proven disease modification. CogniTrace is built to turn a subtle voice signal into an earlier clinical conversation.

The local-first design matters globally. In many places, connectivity is intermittent, clinicians are scarce, and sending raw voice data to a cloud service creates privacy and access barriers. CogniTrace runs the score and the explanation on the phone after setup, using Gemma 4 to make the result understandable without moving sensitive audio off device.

## What I Would Build Next

Next steps are external validation across languages, phones, microphones, and clinical populations; better longitudinal tracking for within-person change; clinician-reviewed explanation templates; and a formal research deployment pathway. I would keep Gemma in the explanation and follow-up role unless future evidence shows a predictive model that beats the ONNX ensemble without increasing false reassurance.

CogniTrace is built on a different premise: the voice may tell part of the story earlier. Early screening should be accessible to any adult 50+ with a smartphone, in any language, without an internet connection. CogniTrace shows that this is technically possible today.
