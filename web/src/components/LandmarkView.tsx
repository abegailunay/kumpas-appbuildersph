import type { ReactNode } from 'react'
import type { LandmarkStream } from '../mp/useLandmarkStream'

/**
 * Video + landmark overlay. MediaPipe always sees the raw, unmirrored frame; for the camera
 * we mirror only the displayed video and canvas together with CSS so the preview feels like a mirror.
 */
export function LandmarkView({ stream, children }: { stream: LandmarkStream; children?: ReactNode }) {
  return (
    <div className="stage">
      <div className={`stage-media${stream.mirrored ? ' mirrored' : ''}`}>
        <video ref={stream.videoRef} playsInline muted />
        <canvas ref={stream.canvasRef} />
      </div>
      {stream.status === 'loading' && <div className="stage-msg">Loading camera and models…</div>}
      {stream.status === 'error' && <div className="stage-msg error">Error: {stream.error}</div>}
      {children}
    </div>
  )
}

export function CountdownOverlay({ count, recording }: { count: number | null; recording: boolean }) {
  if (count !== null) return <div className="countdown">{count}</div>
  if (recording) return <div className="rec-badge">● REC</div>
  return null
}
