import { useEffect, useState } from 'react'
import { Classifier } from '../ml/classifier'

export interface ClipInfo {
  class_id: number
  gloss: string
  file: string
  source: string
  start_s: number
  duration_s: number
}

let classifierPromise: Promise<Classifier> | null = null

export function useClassifier() {
  const [c, setC] = useState<Classifier>()
  useEffect(() => {
    classifierPromise ??= Classifier.load()
    classifierPromise.then(setC, (e) => console.error('model load failed', e))
  }, [])
  return c
}

export function useClipManifest() {
  const [clips, setClips] = useState<ClipInfo[]>([])
  useEffect(() => {
    fetch('/clips/manifest.json')
      .then((r) => (r.ok ? r.json() : { clips: [] }))
      .then((m) => setClips(m.clips))
      .catch(() => setClips([]))
  }, [])
  return clips
}
