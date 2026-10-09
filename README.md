# Kumpas

A browser app that teaches Filipino Sign Language (FSL), with all AI running on-device.

Built for AppBuildersPH Hackathon 2026 (theme: Local AI). See `KUMPAS_SPEC.md` for the build spec.

## Layout

- `ml/` — Python pipeline: feature extraction, GRU training, ONNX export
- `web/` — Vite + React + TypeScript app (camera, MediaPipe, ONNX runtime, Ollama tips)
- `scripts/` — setup scripts

_Status: repo scaffolded; implementation in progress._
