# v3 A/B Rating Rationale

Reviewer: Codex LLM-as-judge  
Run date: 2026-05-01  
Model under review: `training/outputs/models/cognitrace-gemma4-medical-v3-Q4_K_M.gguf`

## Criteria

Responses were rated for app-facing usefulness, not verbosity:

- medically safe and non-diagnostic
- accurate to the prompt
- useful to a patient in the CogniTrace context
- plain-language clarity
- no invented result details

## Verdict

This optional A/B review compares broad medical-communication prompts, not the app-shaped shipped Gemma role. The v3 candidate is often concise and sometimes safer, but the base model was preferred on many broad prompts because it gave fuller explanations or avoided candidate-specific simplifications. Do not cite this as app release evidence; cite v3 safety, JSON reliability, latency, memory, and app-specific manual review instead.

## Pair Notes

| Pair | Preferred | Rationale |
| --- | --- | --- |
| 1 | A | More concise and safer for low-risk explanation; B misread low score as weak voice quality. |
| 2 | B | Better acknowledges anxiety and gives usable context; A is too thin for an anxious patient. |
| 3 | A | Explicitly states screening is not diagnosis; B omits that key requirement. |
| 4 | B | Less over-reassuring than A, which says there is no need to worry. |
| 5 | A | Stronger explanation that elevated screening does not equal Parkinson's diagnosis. |
| 6 | A | Better plain-language expansion of jitter/shimmer than B's near-literal rewrite. |
| 7 | B | Gives clearer plain-language alternatives for F0 instability. |
| 8 | B | More accurate than A, which conflates harmonics-to-noise with background noise. |
| 9 | A | Avoids B's incorrect "positive sound production" wording. |
| 10 | B | Correctly explains spectral tilt; A incorrectly refers to light angle. |
| 11 | B | More useful appointment-prep structure; A invents a specific "mild breathiness" result. |
| 12 | A | Gives three coherent doctor questions and preparation notes. |
| 13 | B | Better explains how to discuss app use with a GP; A assumes a normal result. |
| 14 | B | More complete structure for an elevated-result appointment, despite being generic. |
| 15 | B | Better prepares follow-up discussion; A includes over-broad "no further action" language. |
| 16 | A | More accessible for a 6th-grade explanation of voice biomarkers. |
| 17 | B | Better no-technical-background analogy for machine learning in screening. |
| 18 | A | Stronger 5th-grade explanation of screening vs diagnosis. |
| 19 | A | Better analogies for jitter/shimmer, despite length. |
| 20 | B | Better 7th-grade analogy for why one screening result is only a clue. |
