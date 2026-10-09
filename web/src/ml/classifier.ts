// Runs kumpas.onnx with onnxruntime-web (WASM backend, runtime self-hosted under /ort/).
// The model already contains standardization and softmax: feed features.ts output as-is, never apply norm.json.
import * as ort from 'onnxruntime-web/wasm'
import { F, SEQ_LEN } from './features'

// Only the .wasm is fetched; the JS glue is embedded in the onnxruntime-web/wasm bundle (avoids a dynamic import).
ort.env.wasm.wasmPaths = { wasm: '/ort/ort-wasm-simd-threaded.wasm' }
ort.env.wasm.numThreads = 1

export interface SignClass {
  id: number
  dataset_label_id: number
  gloss: string
  category: string
  english: string
  filipino: string
}

export interface Prediction {
  probs: Float32Array
  ranked: { cls: SignClass; prob: number }[]
  ms: number
}

export class Classifier {
  private session: ort.InferenceSession
  readonly classes: SignClass[]

  private constructor(session: ort.InferenceSession, classes: SignClass[]) {
    this.session = session
    this.classes = classes
  }

  static async load(): Promise<Classifier> {
    const [session, classes] = await Promise.all([
      ort.InferenceSession.create('/models/kumpas.onnx', { executionProviders: ['wasm'] }),
      fetch('/models/classes.json').then((r) => r.json() as Promise<SignClass[]>),
    ])
    return new Classifier(session, classes)
  }

  async predict(features: Float32Array): Promise<Prediction> {
    const t0 = performance.now()
    const out = await this.session.run({ features: new ort.Tensor('float32', features, [1, SEQ_LEN, F]) })
    const probs = out.probs.data as Float32Array
    const ms = performance.now() - t0
    const ranked = [...probs]
      .map((prob, i) => ({ cls: this.classes[i], prob }))
      .sort((a, b) => b.prob - a.prob)
    return { probs, ranked, ms }
  }
}
