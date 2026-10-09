/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  // onnxruntime-web loads its WASM runtime at runtime from /ort/ (self-hosted); don't pre-bundle it.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  test: { environment: 'node' },
})
