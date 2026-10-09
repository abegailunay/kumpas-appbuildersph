// Copies the MediaPipe and ONNX Runtime WASM runtimes into public/ so the app never loads them from a CDN.
import { cpSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const web = join(dirname(fileURLToPath(import.meta.url)), '..')
const nm = join(web, 'node_modules')
const copies = [
  [join(nm, '@mediapipe/tasks-vision/wasm'), join(web, 'public/mediapipe/wasm')],
  ...['ort-wasm-simd-threaded.wasm'].map((f) => [
    join(nm, 'onnxruntime-web/dist', f),
    join(web, 'public/ort', f),
  ]),
]
for (const [src, dst] of copies) {
  mkdirSync(dirname(dst), { recursive: true })
  cpSync(src, dst, { recursive: true })
}
console.log('copied MediaPipe + ONNX Runtime WASM into public/')
