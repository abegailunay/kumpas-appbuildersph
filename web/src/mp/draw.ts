import { HandLandmarker } from '@mediapipe/tasks-vision'
import type { RawFrame } from './landmarker'

const POSE_LINKS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
]
const HAND_LINKS = HandLandmarker.HAND_CONNECTIONS.map((c) => [c.start, c.end] as [number, number])
// Hand colors by MediaPipe label (Right label = signer's right hand on the unmirrored frame).
const HAND_COLORS: Record<number, string> = { 1: '#ffb020', 0: '#22d3ee', [-1]: '#e5e7eb' }

/** Draws in unmirrored frame coordinates; the canvas is mirrored with CSS together with the video. */
export function drawLandmarks(ctx: CanvasRenderingContext2D, f: RawFrame | null) {
  const { width: W, height: H } = ctx.canvas
  ctx.clearRect(0, 0, W, H)
  if (!f) return
  const line = (pts: number[][], links: [number, number][], color: string, w: number) => {
    ctx.strokeStyle = color
    ctx.lineWidth = w
    ctx.beginPath()
    for (const [a, b] of links) {
      ctx.moveTo(pts[a][0] * W, pts[a][1] * H)
      ctx.lineTo(pts[b][0] * W, pts[b][1] * H)
    }
    ctx.stroke()
  }
  const dots = (pts: number[][], color: string, r: number) => {
    ctx.fillStyle = color
    for (const p of pts) {
      ctx.beginPath()
      ctx.arc(p[0] * W, p[1] * H, r, 0, Math.PI * 2)
      ctx.fill()
    }
  }
  if (f.pose_present) {
    line(f.pose, POSE_LINKS, 'rgba(255,255,255,0.7)', 3)
    dots([0, 11, 12, 13, 14, 15, 16].map((i) => f.pose[i]), '#ffffff', 4)
  }
  for (let k = 0; k < 2; k++) {
    if (!f.hand_present[k]) continue
    const color = HAND_COLORS[f.hand_label[k]]
    line(f.hands[k], HAND_LINKS, color, 2.5)
    dots(f.hands[k], color, 3)
  }
}
