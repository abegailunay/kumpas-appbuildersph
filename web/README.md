# web/

Vite + React + TypeScript app. All AI runs in the browser; nothing is loaded from a CDN at runtime.

```bash
npm install      # also copies MediaPipe + ONNX Runtime WASM into public/ (scripts/copy-wasm.mjs)
npm run dev      # http://localhost:5173
npm test         # features.ts vs ml/fixtures/feature_golden.json (must pass)
npm run build
```

Pages: `/debug` (record → top-3 + timings + trim check, `?src=<video url>` to use a clip instead of the camera,
`?delegate=GPU`), `/capture` (record N attempts of a sign, download raw landmarks for `ml/own_samples/`).

- `public/mediapipe/` — `.task` models (committed) + `wasm/` (copied on install)
- `public/ort/` — ONNX Runtime WASM (copied on install)
- `public/models/` — `kumpas.onnx` (standardization + softmax inside; do not apply `norm.json`), `classes.json`, `class_profiles.json`
- `public/clips/` — one trimmed FSL-105 reference clip per sign + `manifest.json` with attribution (made by `ml/make_clips.py`)

`index.html` sets a `connect-src` Content-Security-Policy (same origin + local Ollama only). This also blocks
MediaPipe's built-in usage telemetry to `odml.pa.googleapis.com`.

MediaPipe always receives the raw, unmirrored camera frame; only the preview is mirrored with CSS.
