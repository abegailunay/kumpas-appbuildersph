# Kumpas: agent notes

Spec: `KUMPAS_SPEC.md`. Feature spec (Python + TS must match): `ml/FEATURES.md`.

## Commands
- Python env: `ml/.venv` (Python 3.13, `ml/requirements.txt`). Run scripts from `ml/` with `.venv/bin/python`.
- Extract landmarks: `python extract.py` (resumable, writes `ml/landmarks/`, gitignored)
- Train + report: `python train.py` (MPS; also runs leave-one-signer-out; writes `report.md`)
- Export web model + ONNX parity check: `python export.py`
- Golden fixture for TS features: `python features.py --make-golden`
- Reference clips: `python make_clips.py`
- Web (from `web/`): `npm install`, `npm run dev`, `npm test`, `npm run build`, `npx tsc -b`, `npm run lint`

## Verification
- After changing feature code in either language: `npm test` in `web/` must pass (1e-4 vs Python golden).
- After retraining: `python export.py` asserts ORT == PyTorch.
- Browser end-to-end without a camera: `/debug?src=<webm url>&auto=1`, result in `window.__kumpas.result`.

## Gotchas
- FSL-105 has 4 signers in both official splits; use the LOSO number for "new user" accuracy.
- MediaPipe handedness on unmirrored frames: `Right` label = image-left = signer's right hand.
- Pin npm deps to releases at least 7 days old.
