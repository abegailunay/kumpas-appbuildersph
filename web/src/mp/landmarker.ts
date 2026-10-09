// MediaPipe Tasks (HandLandmarker + PoseLandmarker lite) in VIDEO mode, loaded only from self-hosted files.
// Settings mirror ml/extract.py so live landmarks match the training data.
import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision'

export type Delegate = 'CPU' | 'GPU'

/** One frame of raw landmarks, same layout as one time step of ml/extract.py's .npz. */
export interface RawFrame {
  hands: number[][][] // 2 x 21 x 3, detection order, zeros if absent
  hand_present: number[] // 2
  hand_label: number[] // 2: Left=0, Right=1, -1 absent
  hand_score: number[] // 2
  pose: number[][] // 33 x 4
  pose_present: number
  t_ms: number
  handMs: number
  poseMs: number
}

const HAND_LABELS: Record<string, number> = { Left: 0, Right: 1 }
const zeros = (n: number, m: number) => Array.from({ length: n }, () => new Array<number>(m).fill(0))

export class Landmarkers {
  private lastTs = -1
  readonly hand: HandLandmarker
  readonly pose: PoseLandmarker
  readonly delegate: Delegate

  private constructor(hand: HandLandmarker, pose: PoseLandmarker, delegate: Delegate) {
    this.hand = hand
    this.pose = pose
    this.delegate = delegate
  }

  static async create(delegate: Delegate = 'CPU'): Promise<Landmarkers> {
    const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm')
    const [hand, pose] = await Promise.all([
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: '/mediapipe/hand_landmarker.task', delegate },
        runningMode: 'VIDEO',
        numHands: 2,
      }),
      PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: '/mediapipe/pose_landmarker_lite.task', delegate },
        runningMode: 'VIDEO',
        numPoses: 1,
      }),
    ])
    return new Landmarkers(hand, pose, delegate)
  }

  /** Runs both landmarkers on the raw (unmirrored) video frame. */
  detect(video: HTMLVideoElement): RawFrame {
    const ts = Math.max(this.lastTs + 1, Math.round(performance.now()))
    this.lastTs = ts
    const t0 = performance.now()
    const hr = this.hand.detectForVideo(video, ts)
    const t1 = performance.now()
    const pr = this.pose.detectForVideo(video, ts)
    const t2 = performance.now()

    const frame: RawFrame = {
      hands: [zeros(21, 3), zeros(21, 3)],
      hand_present: [0, 0],
      hand_label: [-1, -1],
      hand_score: [0, 0],
      pose: zeros(33, 4),
      pose_present: 0,
      t_ms: ts,
      handMs: t1 - t0,
      poseMs: t2 - t1,
    }
    hr.landmarks.slice(0, 2).forEach((lms, k) => {
      const cat = hr.handedness[k]?.[0]
      frame.hands[k] = lms.map((p) => [p.x, p.y, p.z])
      frame.hand_present[k] = 1
      frame.hand_label[k] = cat ? (HAND_LABELS[cat.categoryName] ?? -1) : -1
      frame.hand_score[k] = cat?.score ?? 0
    })
    if (pr.landmarks.length) {
      frame.pose = pr.landmarks[0].map((p) => [p.x, p.y, p.z, p.visibility ?? 0])
      frame.pose_present = 1
    }
    return frame
  }

  close() {
    this.hand.close()
    this.pose.close()
  }
}
