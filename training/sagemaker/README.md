# SageMaker Training for Gemma 4 E2B Medical QLoRA

AWS SageMaker training replaces the Kaggle T4 pipeline after Kaggle v7 diverged to NaN (known Gemma 4 + fp16 activation overflow issue). SageMaker on ml.g5.2xlarge (A10G) runs bf16, which removes the fundamental overflow cause.

## Prerequisites

- AWS credentials with SageMaker + S3 access (already configured in this repo)
- `SAGEMAKER_ROLE_ARN` set in the environment, or pass `--role`
- Optional `training/.env` with `HF_TOKEN` for HuggingFace adapter push
- Python venv at `training/sagemaker_venv` with `sagemaker>=2.240,<3` (created by `launch.py` setup)

## Files

| File | Purpose |
|------|---------|
| `src/train.py` | SageMaker entrypoint. Loads Gemma 4 via Unsloth `FastModel`, applies LoRA, trains, saves adapter to `/opt/ml/model`. Includes all fixes from the v7 NaN diagnosis. |
| `src/requirements.txt` | Extra deps installed on top of HF DLC (unsloth 2026.4.6, transformers 5.5.0, xformers). |
| `launch.py` | Local launcher. Uploads data to S3, submits training job via `sagemaker.huggingface.HuggingFace` estimator. |
| `poll_job.sh` | Poll a job until terminal state (Completed / Failed / Stopped). |
| `download_model.sh` | Pull `model.tar.gz` from S3 and extract to `training/outputs/sagemaker_<job>/`. |

## Usage

Smoke test (50 steps, ~10 min, ~$0.30) to verify no NaN:

```bash
cd training
./sagemaker_venv/bin/python sagemaker/launch.py --smoke-test
./sagemaker/poll_job.sh cognitrace-gemma4-smoke-<timestamp>
```

Full training (3 epochs, ~2 hr, ~$2.50):

```bash
./sagemaker_venv/bin/python sagemaker/launch.py
./sagemaker/poll_job.sh cognitrace-gemma4-<timestamp>
```

After completion:

```bash
./sagemaker/download_model.sh cognitrace-gemma4-<timestamp>
# → training/outputs/sagemaker_<job>/ contains adapter_model.safetensors
```

## Fixes Applied (vs Kaggle v7)

Root cause of v7: Gemma architecture hidden-state activations exceed fp16 max (~65k) on T4, causing deterministic NaN. Unsloth has patches but they only apply when loading via `FastModel` / `FastVisionModel`, not `FastLanguageModel`.

Applied in `src/train.py`:

1. **`FastModel` loader** - Gemma 4 multimodal, applies Unsloth's `use_cache` KV-share fix and audio mask clamp.
2. **Explicit audio_tower / vision_tower / multi_modal_projector freeze** - we do text-only, no reason to train these.
3. **`attention_invalid_logits_value = -1e4`** - belt-and-suspenders fp16 overflow guard.
4. **Dataset content converted to list-of-dicts** - Gemma 4 multimodal processor expects `[{"type": "text", "text": "..."}]`.
5. **`chat_template="gemma-4"`** - no thinking mode, straight SFT.
6. **NaN guard callback** - stop training at first NaN log instead of burning 2 hours.
7. **bf16 on A10G** - removes the root cause entirely.
8. **Warmup 50 steps + cosine LR schedule** - gentler ramp-up (v7 used 5 steps + linear).

## Job Outputs

Training job writes to your SageMaker default S3 bucket at `<job-name>/output/model.tar.gz`. Contents after extraction:

```
adapter_config.json
adapter_model.safetensors        # ~124 MB LoRA weights
tokenizer.json / tokenizer_config.json
chat_template.jinja
checkpoint-500/ ...              # intermediate saves (also inside tarball)
trainer_state_summary.json       # final_loss, global_step
```

## Cost Reference

- ml.g5.2xlarge (A10G 24GB, bf16): ~$1.21/hr on-demand
- Smoke test (~15 min including provisioning): ~$0.30
- Full training (~2 hr 15 min): ~$2.70
