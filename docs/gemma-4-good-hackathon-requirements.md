# Gemma 4 Good Hackathon Requirements

Source: https://www.kaggle.com/competitions/gemma-4-good-hackathon  
Captured from Kaggle while logged in: 2026-05-02  
Status note: verify the Kaggle page again before final submission in case organizers update requirements.

## Core Mission

Build a real, working solution that uses Gemma 4 to address a meaningful real-world problem. Kaggle emphasizes positive impact, local/edge usefulness, privacy-sensitive settings, multimodal understanding, post-training, domain adaptation, and grounded outputs.

For model-training submissions, publish the model weights and benchmarks. For app submissions, explain the architecture and demonstrate real-world utility with a functional demo.

## Deadline

- Final submission deadline: May 18, 2026 at 11:59 PM UTC.
- Logged-in Kaggle UI showed the local equivalent as May 18, 2026 at 7:59 PM EDT.
- Draft or unsubmitted writeups are not considered by judges.

## Required Submission Assets

A valid submission must include:

- Kaggle Writeup.
- Attached public video.
- Attached public code repository.
- Attached live demo.
- Media Gallery.

The final submission is made through a Kaggle Writeup. Each team may submit one Writeup, and it can be edited, unsubmitted, and resubmitted before the deadline.

If a private Kaggle resource is attached to a public Writeup, Kaggle says that private resource will automatically become public after the deadline.

## Kaggle Writeup

Purpose: technical proof that the demo is backed by real engineering.

Requirements and expectations:

- Must include title, subtitle, selected track, and detailed analysis.
- Must summarize the project and link supporting resources.
- Should clearly explain the app architecture.
- Should explain exactly how Gemma 4 is used.
- Should describe technical challenges and why the chosen approach was appropriate.
- Maximum length: 1,500 words. Over-limit submissions may be penalized.

## Video

Requirements and expectations:

- Attach the video through the Media Gallery.
- Duration must be 3 minutes or less.
- Video should be published on YouTube.
- Direct YouTube link must be viewable by judges without login.
- It should tell the problem/solution story and demonstrate the project working.

## Public Code Repository

Purpose: source of truth for authenticity and technical validation.

Requirements and expectations:

- Provide a public repository link, such as GitHub or Kaggle Notebook.
- Repository must be accessible without login or paywall.
- Code must be documented.
- Code must clearly show the Gemma 4 implementation.
- Attach the code link under Writeup attachments / Project Links.

## Live Demo

Requirements and expectations:

- Provide either a public URL or demo files for the working project.
- If using a link, attach it under Writeup attachments / Project Links.
- If using files, attach them under Writeup attachments / Files.
- Demo should be publicly accessible without login or paywall when applicable.

## Media Gallery

Requirements and expectations:

- Attach images and/or videos related to the submission.
- A cover image is required to submit the Writeup.

## Relevant Tracks

Main Track:

- Awards best overall projects for vision, technical execution, and real-world impact.

Impact Track:

- Health & Sciences: tools that accelerate discovery or democratize knowledge.
- Global Resilience: offline/edge systems for urgent global challenges.
- Future of Education: adaptive learning and educator support.
- Digital Equity & Inclusivity: linguistic diversity, intuitive interfaces, and AI access.
- Safety & Trust: transparency, reliability, grounding, and explainability.

Special Technology Track:

- Cactus: local-first mobile or wearable app that routes tasks between models.
- LiteRT: compelling use of Google AI Edge LiteRT implementation of Gemma 4.
- llama.cpp: innovative Gemma 4 implementation on resource-constrained hardware.
- Ollama: project that showcases Gemma 4 running locally via Ollama.
- Unsloth: best fine-tuned Gemma 4 model created with Unsloth for a specific impactful task.

## Judging Rubric

Total: 100 points.

| Criterion | Points | What Judges Look For |
| --- | ---: | --- |
| Impact & Vision | 40 | Clear, compelling real-world problem; inspiring vision; tangible positive-change potential. |
| Video Pitch & Storytelling | 30 | Engaging, well-produced video that tells a powerful story. |
| Technical Depth & Execution | 30 | Verified by code and writeup; innovative Gemma 4 use; real, functional, well-engineered technology. |

Kaggle states the video is the primary judging surface, but the writeup and repository are used to verify that the project is functional and built on Gemma 4.

## CogniTrace Implications

Before final submission, CogniTrace should have:

- Public GitHub repository link in the Writeup attachments.
- Public YouTube demo video, 3 minutes or less, showing the actual app flow.
- Public live demo path, likely demo files/TestFlight-equivalent instructions if a web URL is not feasible.
- Media Gallery cover image plus any screenshots/video assets.
- Writeup under 1,500 words, with Health & Sciences selected as the primary Impact Track.
- Architecture explanation covering on-device audio capture, Swift feature extraction, ONNX ensemble scoring, Gemma 4 E2B explanation, GGUF/llama.cpp mobile inference, and privacy/offline behavior.
- Clear statement that Gemma 4 explains and prepares follow-up, while the classical ensemble produces the screening score.
- Public model weights and benchmarks if presenting the fine-tuned Gemma model as part of the submission.
- Benchmarks tied to the exact shipped model artifact, including current v3 safety, JSON reliability, latency, memory, and human/manual review evidence.
- Public code/docs showing the real SageMaker training path if that is the source of the current model, not only the older failed Kaggle notebook.
- Documentation of technical challenges: Gemma raw-audio prediction rejected, mobile memory/GGUF export issues, token/template handling, NaN training instability, and why the final architecture is safer.
