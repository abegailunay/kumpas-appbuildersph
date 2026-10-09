import { useCallback, useEffect, useRef, useState } from 'react'
import type { Recording } from '../ml/sequence'
import { drawLandmarks } from './draw'
import { Landmarkers, type Delegate, type RawFrame } from './landmarker'

export type Source = { kind: 'camera' } | { kind: 'clip'; url: string }

export interface StreamPerf {
  fps: number
  handMs: number
  poseMs: number
  width: number
  height: number
}

// One landmarker pair per delegate for the whole app (survives StrictMode double effects and page switches).
const landmarkerCache = new Map<Delegate, Promise<Landmarkers>>()
export function getLandmarkers(delegate: Delegate): Promise<Landmarkers> {
  if (!landmarkerCache.has(delegate)) landmarkerCache.set(delegate, Landmarkers.create(delegate))
  return landmarkerCache.get(delegate)!
}

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/** 3-2-1 countdown, reporting each number (then null) through `show`. */
export async function countdown(show: (n: number | null) => void, from = 3) {
  for (let n = from; n > 0; n--) {
    show(n)
    await sleep(1000)
  }
  show(null)
}

export function useLandmarkStream(source: Source, delegate: Delegate = 'CPU') {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const listeners = useRef(new Set<(f: RawFrame) => void>())
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [error, setError] = useState<string>()
  const [perf, setPerf] = useState<StreamPerf>({ fps: 0, handMs: 0, poseMs: 0, width: 0, height: 0 })
  const clipUrl = source.kind === 'clip' ? source.url : ''

  useEffect(() => {
    const video = videoRef.current!
    let cancelled = false
    let stream: MediaStream | null = null
    let cb = 0
    setStatus('loading')
    setError(undefined)

    const acc = { n: 0, hand: 0, pose: 0, since: performance.now() }
    const loop = (lm: Landmarkers) => {
      const onFrame = () => {
        if (cancelled) return
        if (video.readyState >= 2 && video.videoWidth) {
          const f = lm.detect(video)
          const canvas = canvasRef.current
          if (canvas) {
            if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth
            if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight
            drawLandmarks(canvas.getContext('2d')!, f)
          }
          listeners.current.forEach((fn) => fn(f))
          acc.n++
          acc.hand += f.handMs
          acc.pose += f.poseMs
          const dt = performance.now() - acc.since
          if (dt > 500) {
            setPerf({ fps: (acc.n * 1000) / dt, handMs: acc.hand / acc.n, poseMs: acc.pose / acc.n,
              width: video.videoWidth, height: video.videoHeight })
            Object.assign(acc, { n: 0, hand: 0, pose: 0, since: performance.now() })
          }
        }
        cb = video.requestVideoFrameCallback(onFrame)
      }
      cb = video.requestVideoFrameCallback(onFrame)
    }

    ;(async () => {
      try {
        const lm = await getLandmarkers(delegate)
        if (source.kind === 'camera') {
          video.removeAttribute('src')
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
            audio: false,
          })
          if (cancelled) return
          video.srcObject = stream
          await video.play()
        } else {
          video.srcObject = null
          video.src = clipUrl
          await new Promise((res, rej) => {
            video.onloadeddata = res
            video.onerror = () => rej(new Error(`cannot load ${clipUrl}`))
          })
        }
        if (cancelled) return
        loop(lm)
        setStatus('ready')
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e))
          setStatus('error')
        }
      }
    })()

    return () => {
      cancelled = true
      video.cancelVideoFrameCallback?.(cb)
      stream?.getTracks().forEach((t) => t.stop())
      video.pause()
    }
  }, [source.kind, clipUrl, delegate])

  const subscribe = useCallback((fn: (f: RawFrame) => void) => {
    listeners.current.add(fn)
    return () => void listeners.current.delete(fn)
  }, [])

  /** Records raw frames: `durationMs` from the camera, or the whole clip from the start for a clip source. */
  const record = useCallback(
    async (durationMs: number): Promise<Recording> => {
      const video = videoRef.current!
      const frames: RawFrame[] = []
      if (source.kind === 'clip') {
        video.pause()
        video.currentTime = 0
        await new Promise((r) => (video.onseeked = r))
        const unsub = subscribe((f) => frames.push(f))
        await video.play()
        await new Promise((r) => (video.onended = r))
        unsub()
      } else {
        const unsub = subscribe((f) => frames.push(f))
        await sleep(durationMs)
        unsub()
      }
      return { frames, width: video.videoWidth, height: video.videoHeight }
    },
    [source.kind, subscribe],
  )

  return { videoRef, canvasRef, status, error, perf, subscribe, record, mirrored: source.kind === 'camera' }
}

export type LandmarkStream = ReturnType<typeof useLandmarkStream>
