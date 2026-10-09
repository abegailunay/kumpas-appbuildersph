# Kumpas — Build Spec (for Claude Code)

AppBuildersPH Hackathon 2026 · Theme: Local AI · Code freeze: **10:00 AM, Oct 10** (submit by 9:30)

## What we're building

Kumpas is a browser app that **teaches Filipino Sign Language (FSL)** and (phase 2) **interprets** it, with all AI running on-device.

- **Phase 1 — Teach (build first):** learner picks a sign → watches a reference clip → tries it on camera → app classifies the attempt locally and says right/wrong with a confidence score and a short tip.
- **Phase 2 — Interpret (only if Teach is done by ~5:00 AM):** user signs a short sequence → app recognizes each sign → local LLM turns the gloss sequence into a natural Filipino sentence → spoken aloud.

Must work with the network off (airplane mode). That is the core demo moment.

## Hard rules (hackathon)

- Fresh repo, built tonight. Do not copy code from other existing projects.
- Repo must be **public** before 10:00 AM.
- Only show **real, measured** numbers (accuracy, latency). No invented benchmarks.
- README must disclose: models, frameworks, dataset, existing assets, AI dev tools (Claude, Claude Code).

## Stack

- **Frontend:** Vite + React + TypeScript. Single page, mobile-first responsive.
- **Hand/pose tracking:** MediaPipe Tasks Vision for web (`@mediapipe/tasks-vision`): HandLandmarker (2 hands) + PoseLandmarker (lite). **Self-host the WASM and `.task` model files in `public/`**. No CDN at runtime, or offline mode breaks.
- **Sign classifier:** small GRU trained in Python (PyTorch), exported to ONNX, run with `onnxruntime-web` (WASM backend; WebGPU optional).
- **Local LLM (tips + Interpret sentences):** Ollama on localhost (`qwen2.5:3b`, or `qwen2.5:1.5b` if RAM is tight), called from the frontend at `http://localhost:11434/api/chat` with `format: "json"`. Configure `OLLAMA_ORIGINS` for the dev origin. **App must still work if Ollama is down** (fall back to template tips and joined glosses).
- **Speech out:** browser `speechSynthesis`. Prefer a Filipino (`fil-PH`/`tl`) voice if installed; otherwise fall back to an English voice and always show text.

## Data: FSL-105

- Source: Mendeley Data `10.17632/48y2y99mb9.2` (also Hugging Face `SEACrowd/fsl_105`). License CC BY 4.0, so attribution is required.
- ~2,130 four-second clips, 105 signs, ~20 clips each, signed by Deaf FSL signers and reviewed by an FSL expert. Includes `train.csv`, `test.csv`, `labels.csv`.
- **First step:** inspect `labels.csv` and the CSV/path format before writing the pipeline. Do not assume column names. Do not invent glosses.

### Class selection
Pick **20–25 signs** from `labels.csv`:
- Prefer greetings, courtesy words, basic questions, family, and common verbs (useful for the demo and Interpret phase).
- Avoid pairs that look alike. After the first training run, drop classes that the confusion matrix shows are confused, and retrain.
- Save the chosen list to `ml/classes.json` (id, gloss, category, English, Filipino meaning if given).

## ML pipeline (`ml/`)

**Critical: feature extraction must be identical in Python and the browser.** Use MediaPipe **Tasks** in Python too (`mediapipe.tasks.python.vision` HandLandmarker + PoseLandmarker in VIDEO mode, same `.task` files), not the legacy `mp.solutions` API. Put the normalization logic in one documented spec and implement it twice (`ml/features.py`, `web/src/ml/features.ts`) with a shared test fixture to confirm both give the same output.

1. `extract.py`: for each clip, run landmarkers per frame and save raw landmarks to `.npy`. Parallelize across CPU cores. Run this first and leave it running while the UI gets built.
2. **Per-frame features:**
   - Pose: shoulders (11, 12), elbows (13, 14), wrists (15, 16), nose (0). Use x,y only.
   - Hands: 21 landmarks × (x, y, z) for left and right, plus a presence flag per hand. Missing hand = zeros, flag 0.
   - Normalize: origin = midpoint of the shoulders, scale = shoulder width. Hand z relative to that hand's wrist.
   - Handedness: use MediaPipe's handedness label. Note that the webcam is mirrored in the UI, so confirm left/right consistency between dataset and live capture.
3. **Sequence:** resample each clip to **32 frames** (linear interpolation over time).
4. **Model:** 2-layer GRU (hidden 128, dropout 0.3) → linear → softmax. Augment with small rotation/scale/translation jitter, time warp, and frame dropout. Do not mirror (it changes handedness).
5. **Train/eval:** use the dataset's train/test split. Report top-1 and top-3 accuracy plus a confusion matrix → `ml/report.md` (these are the real numbers for the pitch).
6. **Fine-tune on the builder's own signing:** the app's Capture mode exports landmark sequences as JSON → `ml/own_samples/`. Mix them in (aim for about 5 per class) and retrain.
7. Export ONNX (opset 17, input `[1, 32, F]`) → `web/public/models/kumpas.onnx` + `classes.json` + `norm.json` (any mean/std used).

## Web app (`web/`)

### Teach mode (Phase 1)
- **Sign library:** grid grouped by category, with search. Each card shows the gloss + meaning + a mastered checkmark.
- **Lesson screen:**
  - Reference clip from FSL-105 (trimmed and converted to `.webm`/`.mp4`, served locally), loopable, with a 0.5× speed toggle. Show a small attribution line under it.
  - Live camera with landmark overlay.
  - **"Try it"** → 3-2-1 countdown → record ~2.5 s of landmarks → resample to 32 frames → classify.
  - **Result:** ✓ if the target is top-1 with prob ≥ threshold (start at 0.6 and tune), else ✗. Show a confidence bar for the target and "Looks like: <top guess>" when wrong.
  - **Tip** (on ✗): rule-based checks comparing the attempt with the class's average features from training (stored in `class_profiles.json`): number of hands used, hand height relative to shoulders/face, amount of movement, which hand is dominant. Send the check results (not raw landmarks) to Ollama to phrase one short, friendly tip in Taglish. If Ollama is unavailable, use the template text.
- **Progress:** mastered after 3 correct attempts. Store in localStorage, with every read/write wrapped in try/catch.
- **Practice quiz:** show the meaning only → user signs it → graded. A good second demo beat.

### Capture mode (dev tool, hidden behind a toggle)
- Choose a class → record N attempts → download a JSON of landmark sequences for fine-tuning.

### Interpret mode (Phase 2, only if time allows)
- Segment signs with a **"hold to sign"** button or spacebar per sign (reliable). Do not attempt continuous segmentation tonight.
- Gloss list → Ollama prompt: "These are FSL glosses in order. Write one natural Filipino sentence and one English sentence. JSON: {fil, en}." → display both + speak the Filipino one.

### Always-visible elements
- **Offline badge** (`navigator.onLine`) + a "Runs on this device" label.
- **Latency panel:** measured landmark ms/frame, classifier ms, LLM ms. Real values from `performance.now()`.

## Repo layout

```
kumpas/
  README.md
  ml/        extract.py, features.py, train.py, export.py, classes.json, report.md, own_samples/
  web/       Vite app; public/mediapipe/, public/models/, public/clips/
  scripts/   setup.sh (or .ps1): installs deps, pulls the Ollama model, downloads MediaPipe assets
```

Don't commit the raw dataset. Do commit the trained `kumpas.onnx` and trimmed reference clips (with attribution), so judges can run the app without retraining.

## README (judges read this)

- One-line pitch + a GIF or screenshot.
- **Why local AI:** privacy of camera and body data, real-time feedback with no network round trip, works offline in classrooms and barangays with poor connectivity, zero per-use cost.
- **What runs locally:** hand/pose tracking, sign classifier, LLM tips/sentences, speech. **What needs the internet:** nothing at runtime (setup downloads only).
- Setup steps for judges (one command if possible) + the hardware it was tested on.
- Measured results from `ml/report.md`.
- Disclosures: models (MediaPipe Hand/Pose Landmarker, custom GRU, Qwen2.5 via Ollama), frameworks, dataset, AI dev tools (Claude, Claude Code).
- **Attribution:** FSL-105 dataset by Isaiah Jassen Tupal and Melvin Cabatuan, De La Salle University, CC BY 4.0. Link to the Mendeley DOI.
- Honest scope note: "Covers N introductory signs. The builder is not a Deaf signer; future work is to co-design with the Deaf community."

## Order of work and checkpoints

| By | Checkpoint |
|---|---|
| 1:00 AM | Dataset downloaded, labels inspected, classes chosen, `extract.py` running in background. Vite app shows camera + landmarks with locally hosted MediaPipe. |
| 2:30 AM | First model trained, `report.md` written, ONNX runs in the browser on a recorded attempt. |
| 4:00 AM | Teach lesson loop works end to end (clip → try → ✓/✗ + confidence). Own samples captured, model fine-tuned. |
| 5:00 AM | Tips (rule-based + LLM), progress, offline badge, latency panel. **Decision point:** start Interpret only if everything above is solid. |
| 6:30 AM | Polish UI, practice quiz, README, repo public. |
| 7:30 AM | Record the ~1-minute demo video (airplane mode on), post on X/LinkedIn tagging Devin/Cognition with #AppBuildersPH. |
| 9:30 AM | Submit on Cerebral Valley. |

## Demo script (Teach-first)

1. Problem: millions of Deaf Filipinos, very few hearing Filipinos who sign.
2. Turn on airplane mode.
3. Pick a sign, watch the clip, sign it badly → ✗ with "Looks like X" and a tip. Sign it correctly → ✓.
4. Practice quiz: sign from the meaning alone.
5. (If built) Interpret: sign 2–3 glosses → spoken Filipino sentence.
6. Show the latency panel and offline badge, then explain why local.
