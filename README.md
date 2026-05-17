# Tzniut Content Classifier

On-device image content filter enforcing Orthodox Jewish (tzniut) modesty standards. Ships as a small (~5MB) classifier for Android and iOS, exported from a Colab-trained multi-task model.

**Critical property:** the model is tuned for **recall over precision on NOT_ACCEPTABLE**. Over-blocking is acceptable; letting an unacceptable image through is not.

---

## Architecture

```
Image
  → MediaPipe person detector (multi-scale, finds all people)
  → For each person crop: MobileNetV4 multi-task classifier
  → Also classifier on full image (cartoon/illustration fallback)
  → Deterministic block rule (OR of any violation across people)
  → BLOCK / ALLOW + reason
```

All training data lives in Cloudflare R2 + HuggingFace Datasets. Training runs on Google Colab / Kaggle / Lightning AI free tiers. The only thing on your laptop is the orchestrator CLI, the human review UI, and the final model file.

See [plan](../.claude/plans/you-are-the-best-tidy-glade.md) for full design.

---

## Setup

### 1. Accounts (all free, no credit card)

| Service | Why | URL |
|---------|-----|-----|
| Google | Drive, Colab, Gemini | already have |
| HuggingFace | Datasets storage + streaming | https://huggingface.co/join |
| Cloudflare | R2 bulk image storage (unlimited egress) | https://dash.cloudflare.com/sign-up |
| NVIDIA NIM | VLM labeling, ~40 RPM free | https://build.nvidia.com |
| Google AI Studio | Gemini API key | https://aistudio.google.com/apikey |
| Kaggle | Backup free GPU (30h/week P100) | https://www.kaggle.com |
| Lightning AI | Backup free GPU (80h/month) | https://lightning.ai |
| Unsplash | Image source API | https://unsplash.com/developers |
| Pexels | Image source API | https://www.pexels.com/api |

### 2. Local install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/.env.example .env
# Edit .env with your API keys + bucket names
```

### 3. Smoke test

```bash
python scripts/smoke_test.py
```
Verifies R2, HF, NIM, and Gemini connectivity.

---

## Run order

The pipeline is sequential. Each step is fully resumable — if a Colab session dies, restart the same cell and it picks up where it left off.

| Step | Where it runs | What it does | Typical time |
|------|---------------|--------------|--------------|
| 1 | Colab: `notebooks/01_collection.ipynb` | Pull 100-200k images from open sources → R2 | 1-3 days (parallel) |
| 2 | Colab: `notebooks/02_labeling.ipynb` | NSFW oracle + VLM labels → HuggingFace Dataset | 3-7 days (rate-limited) |
| 3 | Laptop: `python -m review_ui.server` | Human-confirm uncertain labels (seed round) | 5-10 hours |
| 4 | Colab: `notebooks/03_training.ipynb` | Train multi-task MobileNetV4 | 4-12 hours |
| 5 | Colab: `notebooks/04_export.ipynb` | Export `.tflite` (Android), `.mlpackage` (iOS), `.keras` (full) | ~10 min |
| 6 | Colab: `notebooks/05_active_learning.ipynb` | Sample uncertain unlabeled images, repeat 2-5 | weekly cadence |

Final models land at `models/tzniut.tflite`, `models/tzniut.mlpackage`, `models/tzniut.keras`.

---

## Project structure

```
config/                  master YAML config, threshold tables, env template
prompts/                 halachic labeling prompt (single most important file)
pipelines/
  collection/            one collector per source + dedup + R2 client
  labeling/              Falconsai NSFW oracle, NIM/Gemini VLM labeler, rate limiter
  training/              dataset stream, multi-task model, train, calibration, thresholds
  export/                TFLite + CoreML + Keras exporters, calibration set builder
  eval/                  benchmark, HTML visual report, on-device latency
  active_learning/       uncertainty sampling, round runner
notebooks/               Colab orchestrators (one per phase)
review_ui/               local FastAPI + HTML/JS, streams from R2 one image at a time
manifests/               parquet manifests for collection, labels, human review
test_sets/               versioned holdout test set manifest
models/                  final exported model files (only thing on laptop)
scripts/                 smoke tests + small one-off utilities
orchestrator.py          local CLI to kick off + monitor Colab jobs
```

---

## Limitations

Read [the plan](../.claude/plans/you-are-the-best-tidy-glade.md#limitations--honest-disclosures) for the full list. The big ones:

- Background figures <20px tall: best-effort, not guaranteed
- Cartoons/illustrations: 5-10% lower recall than photos
- Text-only billboards (no human shape): out of scope for v1
- API rate limits can shift — the config has knobs

---

## License + ethics

- All shipped model weights are Apache 2.0 (Falconsai NSFW, MediaPipe, MobileNetV4 backbone)
- No commercial-license blockers (deliberately avoided YOLOv11n / AGPL components)
- We **never** source or seek out partially-clothed children. NOT_ACCEPTABLE patterns train on adults; the model generalizes to children since halachic standards are identical across ages.
