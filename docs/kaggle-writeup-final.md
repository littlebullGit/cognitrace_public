# CogniTrace: 60-Second On-Device Parkinson’s Voice Screening with Gemma 4

**Subtitle:** An iPhone prototype that turns voice biomarkers into private, on-device screening guidance.

**Tracks:** Health & Sciences (Main) | llama.cpp & Unsloth (Special Technology)

## Project Links & Demo Access

* **Public Repository:** [github.com/littlebullGit/cognitrace_public](https://github.com/littlebullGit/cognitrace_public)
* **Model Weights (GGUF):** [huggingface.co/littlebull9/cognitrace-gemma4-medical-GGUF](https://huggingface.co/littlebull9/cognitrace-gemma4-medical-GGUF)
* **Video Pitch:** [youtu.be/tUi3j9qtI3g](https://youtu.be/tUi3j9qtI3g)
* **Demo Access:** The live demo is available through Apple TestFlight: [testflight.apple.com/join/jE22f3b7](https://testflight.apple.com/join/jE22f3b7). Because CogniTrace is an edge-only iOS app, there is no web demo. The full workflow is shown in the video pitch above.

## The Story

Growing up, I could always find my grandfather on the sideline. Soccer, hockey, piano recitals, family gatherings: he was usually there with a camera, recording the moments the rest of us were too busy living through. He filmed goals, performances, birthdays, and ordinary afternoons, then turned them into the family highlight reel.

That changed when he was 76. First, the footage became shaky. Then the camera came out less often. His hands trembled too much to hold it steady, and walking to the edge of the field became harder. He was later diagnosed with Parkinson’s disease.

Years later, while studying computational biology, I learned that speech and voice changes can appear years before the motor symptoms that usually lead someone to a clinic. By the time my grandfather’s tremor made the disease visible, an earlier window for noticing subtle changes may already have passed.

CogniTrace began with a simple question: if early signals can appear in the voice, why should the first screen depend on specialist access, clinic wait times, or reliable connectivity?

## The Problem

Parkinson’s screening has an access problem. Specialist care can be expensive, unevenly distributed, and slow to reach. Voice is different. It is easy to capture, repeatable, and available through a device many people already carry.

But a screening tool should not give users only a number. A percentage score can scare someone, falsely reassure them, or leave them unsure what to do next. CogniTrace is designed to explain what the voice check found, what it cannot know, and what the user can discuss with a clinician.

## What CogniTrace Does

CogniTrace is a working iPhone prototype for adults 50 and older. The user completes three short voice tasks: a sustained vowel, rapid syllables, and free speech. The app extracts 56 acoustic biomarkers on device using Swift, vDSP, and Accelerate. It then runs an ONNX Runtime stacked ensemble of LightGBM, XGBoost, and CatBoost. The classifier returns a low, moderate, or elevated screening result in under 2 ms.

Gemma 4 has a separate role. The ensemble scores the voice check. Gemma explains the result, translates it, teaches the relevant biomarkers, and helps the user prepare for a doctor visit. The prompt is grounded in the current score, risk band, and biomarker context, so the assistant stays tied to the user’s actual screening state instead of giving broad medical advice.

The app is local-first. The roughly 2.7 GB Gemma model downloads in the background, so recording and ONNX scoring can run immediately. Once the model is downloaded, Gemma interpretation runs offline. Audio history stays on the device and can be deleted locally.

## Prior Work and the Gap

Voice-based Parkinson’s screening is an active research area. Sage Bionetworks’ mPower study showed that phones can collect Parkinson-related voice and movement signals at scale through Apple ResearchKit. Academic groups have studied speech changes over time, and commercial voice-biomarker platforms already exist.

CogniTrace focuses on a narrower product gap: a consumer-facing, on-device workflow that combines voice screening with an LLM explanation layer. Many existing systems are cloud-based, research-only, or built primarily for clinicians. CogniTrace explores what this could look like as a private, phone-based workflow for the user.

## Why This Uses Gemma 4

In a health screening app, the risky moment is the transition from a score to a human decision. A score can confuse people or be taken more seriously than the evidence supports. CogniTrace uses Gemma 4 E2B as the interpretation layer between acoustic biomarkers and the user’s next step.

Gemma 4 provides:

* Plain-language summaries for low, moderate, and elevated results.
* Multilingual explanations in English, Italian, Chinese, Spanish, and French.
* Biomarker education for features such as jitter, shimmer, pitch stability, and syllable timing.
* Doctor-visit preparation, including questions to ask and information to bring.
* Guardrails that keep the assistant focused on screening education and follow-up preparation.

I also tested a more ambitious approach: using Gemma as a raw-audio predictor. That experiment did not meet the product standard. Native-audio Gemma performed near chance on this dataset and introduced too much risk of false reassurance. The final architecture keeps prediction in the validated classifier and uses Gemma where it adds clarity, language access, and safer follow-up guidance.

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

The classifier is trained on the same Swift-extracted feature vectors used by the app, so mobile inference matches the training feature contract. The Gemma runtime uses GGUF Q4_K_M quantization through llama.cpp with Metal acceleration.

To test edge viability on older hardware, I ran the app on iPhone 13, 14, and 15 devices. On an iPhone 14, the classifier result appears almost immediately, while the first Gemma summary takes about 15 seconds.

This split is intentional. Health screening needs clear model roles. CogniTrace keeps the deterministic scoring path separate from generated text. Gemma receives a constrained prompt containing only the current score, risk band, relevant biomarkers, and safety instructions.

## Training Path and Proof Artifact

Fine-tuning was necessary because the zero-shot base model did not reliably produce the rigid JSON structure required by the Swift app. It also sometimes drifted into generic medical advice. Fine-tuning helped the model follow the required output format, stay within the safety guardrails, and ground its response in the user’s acoustic screening state.

The shipped model uses a SageMaker and Unsloth training path. After early Kaggle fp16 training runs diverged to NaN, I trained with bf16 on SageMaker. I preserved chat-template formatting, froze multimodal towers, added NaN guards, used cosine scheduling, and exported the exact local GGUF loaded by the app: `cognitrace-gemma4-medical-v3-Q4_K_M.gguf`.

I reran evaluation against that exact v3 artifact. The app-facing gates passed:

* 0/25 adversarial safety violations across English, Italian, Chinese, Spanish, and French.
* 30/30 valid generated responses in the reliability check.
* First-token latency within 1.034x of the base GGUF.
* Peak memory only +29 MB over the base model in local GGUF evaluation.
* Manual review covering initial summaries, moderate follow-up, biomarker explanation, treatment refusal, and Chinese output.

## Classifier Validation

The voice classifier was evaluated with subject-grouped 5-fold cross-validation on 799 Swift-extracted recordings from 59 subjects. Recordings from the same subject never appeared in both the training and validation folds. The current internal result is 95.74% accuracy, 0.9574 macro-F1, and 0.9908 AUC-ROC.

These results are promising, but the scope is limited. CogniTrace is a screening prototype, not a diagnostic tool. It was trained on one open Italian Parkinson voice corpus. Real-world deployment would require external validation across languages, phones, microphones, rooms, fatigue states, respiratory illness, and broader clinical populations.

## Impact

CogniTrace is designed for the gap between “I wonder if something is changing” and “I can get a specialist appointment.” A private voice check on a phone cannot replace a neurologist. It can help users notice patterns, save prior checks, delete their data, and walk into a clinical conversation with clearer questions.

That timing matters. CogniTrace cannot prevent Parkinson’s disease, but earlier screening can support earlier clinical conversations. Users may seek care sooner, when evidence-backed interventions and monitoring may be more useful. For example, high-intensity treadmill exercise has been studied in newly diagnosed Parkinson’s and is associated with less motor-score worsening over six months.

The local-first design also matters globally. In many places, connectivity is intermittent, clinicians are scarce, and sending raw voice data to the cloud creates privacy and access barriers. CogniTrace runs the score and the explanation on the phone after setup, using Gemma 4 to make the result understandable without moving sensitive audio off the device.

## What I Would Build Next

Next steps include external validation across languages, phones, microphones, and clinical populations; better longitudinal tracking for within-person change; clinician-reviewed explanation templates; and a formal research deployment pathway. I would keep Gemma in the explanation and follow-up role unless future evidence shows that a generative predictive model can outperform the ONNX ensemble without increasing false reassurance.

CogniTrace is built on one premise: the voice may reveal part of the story earlier. Early screening should be accessible to any adult 50+ with a smartphone, in any language, without requiring an internet connection. CogniTrace shows that this workflow is technically possible today.
