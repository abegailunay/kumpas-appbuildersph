# Kumpas feature spec (version 1)

This is the single source of truth for turning raw MediaPipe landmarks into the
classifier input. It is implemented twice and the two must agree:

- Python: `ml/features.py` (used for training and export)
- Browser: `web/src/ml/features.ts` (used live)

`ml/fixtures/feature_golden.json` holds raw inputs and the exact expected
outputs. The TypeScript implementation must match `expected.features` within an
absolute tolerance of `1e-5` per value.

All arithmetic is float64. Only the final `(32, 142)` output is cast to float32.

## 1. Raw input (one recorded sequence of T frames)

Produced by MediaPipe Tasks `HandLandmarker` (`num_hands=2`) and
`PoseLandmarker` lite (`num_poses=1`), both in VIDEO mode, using the `.task`
files in `web/public/mediapipe/`. The frame given to MediaPipe must be
**unmirrored** (the raw camera image). Mirroring the camera preview is a
display concern only (CSS), never applied to the pixels sent to MediaPipe.

| Field | Shape | Meaning |
|---|---|---|
| `hands` | T × 2 × 21 × 3 | normalized `x, y` and MediaPipe `z`, slots in MediaPipe detection order, zeros if absent |
| `hand_present` | T × 2 | 1 if a hand is in that slot |
| `hand_label` | T × 2 | MediaPipe handedness `category_name`: `Left`=0, `Right`=1, absent/unknown = -1 |
| `pose` | T × 33 × 4 | `x, y, z, visibility` of all pose landmarks |
| `pose_present` | T | 1 if a pose was detected |
| `width`, `height` | scalars | pixel size of the frames given to MediaPipe |

Only hand slots 0 and 1 are used (the first two detections).

Empirically (on unmirrored FSL-105 video) the hand labelled `Right` appears on
the **image-left** side, i.e. it is the signer's anatomical right hand.

## 2. Constants

```
SEQ_LEN        = 32
PAD            = 5
POSE_IDX       = [0, 11, 12, 13, 14, 15, 16]   # nose, shoulders, elbows, wrists
LEFT = 0, RIGHT = 1                            # MediaPipe handedness values
FALLBACK_SCALE = 0.3
F              = 142
```

`median(v)`: sort ascending; odd length → middle value; even length → mean of
the two middle values.

## 3. Aspect correction

`a = width / height`. Every image `x` (pose and hand) and every hand `z` is
multiplied by `a` before use, so x, y and z are all in units of image height.
This makes features independent of camera aspect ratio (16:9 dataset vs. e.g.
4:3 webcam).

## 4. Trim

`any[t] = hand_present[t][0] == 1 or hand_present[t][1] == 1`.

- If no frame has a hand: `start = 0`, `end = T - 1`.
- Else with `first`/`last` the first/last `t` where `any[t]`:
  `start = max(0, first - PAD)`, `end = min(T - 1, last + PAD)`.

Keep frames `start..end` inclusive. `N = end - start + 1`. All later steps use
only kept frames, re-indexed from 0.

## 5. Body frame (one per sequence)

Over kept frames with `pose_present == 1`, with `L = pose[t][11]`,
`R = pose[t][12]`:

```
mid_x[t] = ((L.x + R.x) / 2) * a
mid_y[t] = (L.y + R.y) / 2
sw[t]    = hypot((L.x - R.x) * a, L.y - R.y)
ox = median(mid_x),  oy = median(mid_y),  s = median(sw)
```

If no kept frame has a pose: `ox = a / 2`, `oy = 0.5`, `s = FALLBACK_SCALE`.
If `s <= 1e-6`: `s = FALLBACK_SCALE`.

`(ox, oy)` is the shoulder midpoint; the vertical line `x = ox` is the
**shoulder midline**.

## 6. Handedness resolution → sides

For each kept frame, let `D` be the slots `k ∈ {0, 1}` (in order) with
`hand_present[t][k] == 1`. Each detected hand is assigned a side
`RIGHT` or `LEFT`. Wrist = landmark 0.

- `|D| == 1` (slot `k`): side = `hand_label[t][k]` if it is 0 or 1. Otherwise
  side = `RIGHT` if `hands[t][k][0].x * a < ox` else `LEFT`.
- `|D| == 2`: if the two labels differ and both are 0 or 1, each hand takes its
  label. Otherwise (same label, or a label is -1) assign by x-position relative
  to the shoulder midline: slot 0 gets `RIGHT` if
  `hands[t][0][0].x <= hands[t][1][0].x` (it is further toward image-left of the
  midline), else `LEFT`; slot 1 gets the other side.

Result: `side_hand[t][s]` (21 × 3) and `side_present[t][s]` for `s ∈ {LEFT, RIGHT}`.

## 7. Dominant / non-dominant slots

`nR = Σ side_present[t][RIGHT]`, `nL = Σ side_present[t][LEFT]`.
`dominant = RIGHT if nR >= nL else LEFT`; `nondominant` is the other side.
This is decided once per sequence. No coordinates are mirrored.

## 8. Per-frame features (142 values)

For kept frame `t`:

| Index | Content |
|---|---|
| 0–13 | for `p` in `POSE_IDX` (in order): `(pose[t][p].x * a - ox) / s`, `(pose[t][p].y - oy) / s` |
| 14–76 | dominant hand, for `j = 0..20`: `(x*a - ox)/s`, `(y - oy)/s`, `(z_j - z_0) * a / s` |
| 77 | dominant hand present flag (1 or 0) |
| 78–140 | non-dominant hand, same layout |
| 141 | non-dominant hand present flag |

`z_0` is the wrist `z` of the same hand in the same frame.
If `pose_present[t] == 0`, indices 0–13 are all 0.
If a hand side is absent in frame `t`, its 63 coordinates and its flag are all 0.

Useful offsets: dominant wrist x/y = 14/15, non-dominant wrist x/y = 78/79,
nose x/y = 0/1. Positive y is **below** the shoulder line (image y grows down).

## 9. Resample to 32 frames

Linear interpolation over time, applied to all 142 values (flags included, so
they may become fractional):

```
for j in 0..31:
  pos = (j * (N - 1)) / 31
  i0  = floor(pos);  i1 = min(i0 + 1, N - 1);  w = pos - i0
  out[j] = f[i0] * (1 - w) + f[i1] * w
```

If `N == 1`, every output frame equals `f[0]`.

Output: `(32, 142)` cast to float32. This is fed as-is to the ONNX model
(input `features`, shape `[1, 32, 142]`). Per-feature standardization
(`norm.json`) is already **inside** the ONNX graph, so the browser must not
apply it again.

## 10. Attempt stats (for Teach tips)

Computed from the `(32, 142)` output; `class_profiles.json` holds per-class
mean/std of the same stats over the training clips. For `hand ∈ {dom, nondom}`
with offset `o` (14 or 78) and flag `fl[j] = out[j][o + 63]`, `on[j] = fl[j] >= 0.5`:

- `<hand>_presence` = mean of `fl` over 32 frames
- `<hand>_wrist_y` = mean of `out[j][o + 1]` over frames where `on[j]`; `null` if none.
  In shoulder widths below the shoulder line (negative = above).
- `<hand>_movement` = Σ over `j = 1..31` where `on[j] and on[j-1]` of
  `hypot(out[j][o] - out[j-1][o], out[j][o+1] - out[j-1][o+1])` (wrist path length in shoulder widths)
- `hands_used` = `(dom_presence >= 0.3) + (nondom_presence >= 0.3)` (0, 1 or 2)
- `nose_y` = mean of `out[j][1]` over frames where indices 0–13 are not all 0; `null` if none.

`dominant` (`"right"`/`"left"`, step 7) is reported alongside, not derived from the features.
