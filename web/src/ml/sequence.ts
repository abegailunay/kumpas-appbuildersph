import type { RawFrame } from '../mp/landmarker'
import type { RawSequence } from './features'

export interface Recording {
  frames: RawFrame[]
  width: number
  height: number
}

export const toRawSequence = ({ frames, width, height }: Recording): RawSequence => ({
  width,
  height,
  hands: frames.map((f) => f.hands),
  hand_present: frames.map((f) => f.hand_present),
  hand_label: frames.map((f) => f.hand_label),
  pose: frames.map((f) => f.pose),
  pose_present: frames.map((f) => f.pose_present),
})

export function measuredFps(frames: RawFrame[]): number {
  if (frames.length < 2) return 0
  return ((frames.length - 1) * 1000) / (frames[frames.length - 1].t_ms - frames[0].t_ms)
}

/**
 * One attempt in exactly the key layout of ml/extract.py's .npz files
 * (hands, hand_present, hand_label, hand_score, pose, pose_present, timestamps_ms,
 * label_id, split, fps, width, height, src), so fine-tuning can load it like an extracted clip.
 */
export function toExtractRecord(rec: Recording, clipId: string, labelId: number) {
  const t0 = rec.frames[0]?.t_ms ?? 0
  return {
    clip_id: clipId,
    ...toRawSequence(rec),
    hand_score: rec.frames.map((f) => f.hand_score),
    timestamps_ms: rec.frames.map((f) => f.t_ms - t0),
    label_id: labelId,
    split: 'own',
    fps: measuredFps(rec.frames),
    src: 'capture',
  }
}

export function downloadJson(name: string, data: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' }))
  const a = Object.assign(document.createElement('a'), { href: url, download: name })
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
