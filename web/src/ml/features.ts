// Feature extraction, a line-by-line port of ml/features.py implementing ml/FEATURES.md (spec version 1).
// Tested against ml/fixtures/feature_golden.json in features.test.ts. Keep both implementations in sync.

export const SPEC_VERSION = 1
export const SEQ_LEN = 32
export const PAD = 5
export const POSE_IDX = [0, 11, 12, 13, 14, 15, 16] as const
export const LEFT = 0
export const RIGHT = 1
export const FALLBACK_SCALE = 0.3
export const N_POSE = 2 * POSE_IDX.length
export const HAND_BLOCK = 21 * 3 + 1
export const DOM_OFF = N_POSE
export const NONDOM_OFF = N_POSE + HAND_BLOCK
export const F = N_POSE + 2 * HAND_BLOCK

/** Raw MediaPipe output for one sequence (FEATURES.md section 1). Unmirrored frame coordinates. */
export interface RawSequence {
  width: number
  height: number
  hands: number[][][][] // T x 2 x 21 x 3, detection order, zeros if absent
  hand_present: number[][] // T x 2
  hand_label: number[][] // T x 2: Left=0, Right=1, absent/unknown=-1
  pose: number[][][] // T x 33 x 4 (x, y, z, visibility)
  pose_present: number[] // T
}

export interface FeatureMeta {
  trim: [number, number]
  origin: [number, number]
  scale: number
  dominant: 'right' | 'left'
}

export interface AttemptStats {
  dom_presence: number
  dom_wrist_y: number | null
  dom_movement: number
  nondom_presence: number
  nondom_wrist_y: number | null
  nondom_movement: number
  hands_used: number
  nose_y: number | null
}

/** numpy-compatible median: mean of the two middle values for even lengths. */
export function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

export function trimRange(handPresent: number[][]): [number, number] {
  const T = handPresent.length
  let first = -1
  let last = -1
  for (let t = 0; t < T; t++) {
    if (handPresent[t][0] || handPresent[t][1]) {
      if (first < 0) first = t
      last = t
    }
  }
  if (first < 0) return [0, T - 1]
  return [Math.max(0, first - PAD), Math.min(T - 1, last + PAD)]
}

export function bodyFrame(pose: number[][][], posePresent: number[], a: number): [number, number, number] {
  const midX: number[] = []
  const midY: number[] = []
  const sw: number[] = []
  for (let t = 0; t < pose.length; t++) {
    if (!posePresent[t]) continue
    const L = pose[t][11]
    const R = pose[t][12]
    midX.push(((L[0] + R[0]) / 2) * a)
    midY.push((L[1] + R[1]) / 2)
    sw.push(Math.hypot((L[0] - R[0]) * a, L[1] - R[1]))
  }
  if (!midX.length) return [a / 2, 0.5, FALLBACK_SCALE]
  const s = median(sw)
  return [median(midX), median(midY), s > 1e-6 ? s : FALLBACK_SCALE]
}

/** Returns side-indexed hands [t][side] (side = LEFT/RIGHT) and presence. */
export function assignSides(
  hands: number[][][][],
  handPresent: number[][],
  handLabel: number[][],
  a: number,
  ox: number,
): { side: (number[][] | null)[][]; present: boolean[][] } {
  const side: (number[][] | null)[][] = []
  const present: boolean[][] = []
  for (let t = 0; t < hands.length; t++) {
    const sd: (number[][] | null)[] = [null, null]
    const pr = [false, false]
    const det = [0, 1].filter((k) => handPresent[t][k])
    if (det.length === 1) {
      const k = det[0]
      const lab = handLabel[t][k]
      const s = lab === LEFT || lab === RIGHT ? lab : hands[t][k][0][0] * a < ox ? RIGHT : LEFT
      sd[s] = hands[t][k]
      pr[s] = true
    } else if (det.length === 2) {
      const l0 = handLabel[t][0]
      const l1 = handLabel[t][1]
      const valid = (l: number) => l === LEFT || l === RIGHT
      const s0 = l0 !== l1 && valid(l0) && valid(l1) ? l0 : hands[t][0][0][0] <= hands[t][1][0][0] ? RIGHT : LEFT
      sd[s0] = hands[t][0]
      sd[1 - s0] = hands[t][1]
      pr[0] = pr[1] = true
    }
    side.push(sd)
    present.push(pr)
  }
  return { side, present }
}

/** Steps 3-8: raw sequence -> N x F per-frame features (float64) + meta. */
export function frameFeatures(raw: RawSequence): { frames: number[][]; meta: FeatureMeta } {
  const a = raw.width / raw.height
  const [start, end] = trimRange(raw.hand_present)
  const hands = raw.hands.slice(start, end + 1)
  const hp = raw.hand_present.slice(start, end + 1)
  const hl = raw.hand_label.slice(start, end + 1)
  const pose = raw.pose.slice(start, end + 1)
  const pp = raw.pose_present.slice(start, end + 1)

  const [ox, oy, s] = bodyFrame(pose, pp, a)
  const { side, present } = assignSides(hands, hp, hl, a, ox)
  let nR = 0
  let nL = 0
  for (const p of present) {
    nR += +p[RIGHT]
    nL += +p[LEFT]
  }
  const dom = nR >= nL ? RIGHT : LEFT

  const frames: number[][] = []
  for (let t = 0; t < hands.length; t++) {
    const f = new Array<number>(F).fill(0)
    if (pp[t]) {
      POSE_IDX.forEach((p, i) => {
        f[2 * i] = (pose[t][p][0] * a - ox) / s
        f[2 * i + 1] = (pose[t][p][1] - oy) / s
      })
    }
    for (const [off, sd] of [
      [DOM_OFF, dom],
      [NONDOM_OFF, 1 - dom],
    ] as const) {
      const h = side[t][sd]
      if (!present[t][sd] || !h) continue
      const z0 = h[0][2]
      for (let j = 0; j < 21; j++) {
        f[off + 3 * j] = (h[j][0] * a - ox) / s
        f[off + 3 * j + 1] = (h[j][1] - oy) / s
        f[off + 3 * j + 2] = ((h[j][2] - z0) * a) / s
      }
      f[off + 63] = 1
    }
    frames.push(f)
  }
  return { frames, meta: { trim: [start, end], origin: [ox, oy], scale: s, dominant: dom === RIGHT ? 'right' : 'left' } }
}

/** Step 9: linear resample over time to n frames. */
export function resample(seq: number[][], n = SEQ_LEN): number[][] {
  const N = seq.length
  const out: number[][] = []
  for (let j = 0; j < n; j++) {
    const pos = (j * (N - 1)) / (n - 1)
    const i0 = Math.min(Math.max(Math.floor(pos), 0), N - 1)
    const i1 = Math.min(i0 + 1, N - 1)
    const w = pos - i0
    out.push(seq[i0].map((v, k) => v * (1 - w) + seq[i1][k] * w))
  }
  return out
}

/** Full pipeline: raw -> flat Float32Array of SEQ_LEN * F (row-major [frame][feature]), ready for the model. */
export function extractFeatures(raw: RawSequence): { features: Float32Array; meta: FeatureMeta; nFrames: number } {
  const { frames, meta } = frameFeatures(raw)
  return { features: Float32Array.from(resample(frames).flat()), meta, nFrames: frames.length }
}

/** Step 10: stats from a flat SEQ_LEN * F feature array. */
export function attemptStats(x: Float32Array | number[]): AttemptStats {
  const at = (j: number, k: number) => x[j * F + k]
  const st: Record<string, number | null> = {}
  for (const [name, off] of [
    ['dom', DOM_OFF],
    ['nondom', NONDOM_OFF],
  ] as const) {
    let flagSum = 0
    let ySum = 0
    let nOn = 0
    let move = 0
    for (let j = 0; j < SEQ_LEN; j++) {
      const fl = at(j, off + 63)
      flagSum += fl
      if (fl >= 0.5) {
        ySum += at(j, off + 1)
        nOn++
        if (j > 0 && at(j - 1, off + 63) >= 0.5)
          move += Math.hypot(at(j, off) - at(j - 1, off), at(j, off + 1) - at(j - 1, off + 1))
      }
    }
    st[`${name}_presence`] = flagSum / SEQ_LEN
    st[`${name}_wrist_y`] = nOn ? ySum / nOn : null
    st[`${name}_movement`] = move
  }
  let noseSum = 0
  let nPose = 0
  for (let j = 0; j < SEQ_LEN; j++) {
    let any = false
    for (let k = 0; k < N_POSE; k++) any ||= at(j, k) !== 0
    if (any) {
      noseSum += at(j, 1)
      nPose++
    }
  }
  st.hands_used = +((st.dom_presence as number) >= 0.3) + +((st.nondom_presence as number) >= 0.3)
  st.nose_y = nPose ? noseSum / nPose : null
  return st as unknown as AttemptStats
}
