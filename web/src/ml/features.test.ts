import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { attemptStats, extractFeatures, F, SEQ_LEN, SPEC_VERSION, type RawSequence } from './features'

const TOL = 1e-4
const golden = JSON.parse(
  readFileSync(fileURLToPath(new URL('../../../ml/fixtures/feature_golden.json', import.meta.url)), 'utf8'),
)

describe('features.ts matches ml/features.py (feature_golden.json)', () => {
  it('uses the same spec version and shapes', () => {
    expect(golden.spec_version).toBe(SPEC_VERSION)
    expect(golden.feature_dim).toBe(F)
    expect(golden.seq_len).toBe(SEQ_LEN)
  })

  for (const c of golden.cases) {
    it(c.name, () => {
      const { features, meta } = extractFeatures(c.input as RawSequence)
      const exp = c.expected
      expect(meta.trim).toEqual(exp.trim)
      expect(meta.dominant).toBe(exp.dominant)
      expect(meta.scale).toBeCloseTo(exp.scale, 10)
      expect(meta.origin[0]).toBeCloseTo(exp.origin[0], 10)
      expect(meta.origin[1]).toBeCloseTo(exp.origin[1], 10)

      const flat = (exp.features as number[][]).flat()
      expect(features.length).toBe(flat.length)
      let maxDiff = 0
      let worst = -1
      flat.forEach((v, i) => {
        const d = Math.abs(features[i] - v)
        if (d > maxDiff) [maxDiff, worst] = [d, i]
      })
      console.log(`${c.name}: max |diff| = ${maxDiff.toExponential(2)} (frame ${Math.floor(worst / F)}, feature ${worst % F})`)
      expect(maxDiff).toBeLessThan(TOL)

      const st = attemptStats(features)
      for (const [k, v] of Object.entries(exp.stats as Record<string, number | null>)) {
        const got = st[k as keyof typeof st]
        if (v === null) expect(got).toBeNull()
        else expect(Math.abs((got as number) - v)).toBeLessThan(TOL)
      }
    })
  }
})
