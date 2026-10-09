"""Feature extraction, implementing ml/FEATURES.md exactly (spec version 1).

The browser port (web/src/ml/features.ts) must match this file; the golden
fixture ml/fixtures/feature_golden.json is generated here:

  python features.py --make-golden
"""

import argparse
import json
from pathlib import Path

import numpy as np

SPEC_VERSION = 1
SEQ_LEN = 32
PAD = 5
POSE_IDX = (0, 11, 12, 13, 14, 15, 16)
LEFT, RIGHT = 0, 1
FALLBACK_SCALE = 0.3
N_POSE = 2 * len(POSE_IDX)
HAND_BLOCK = 21 * 3 + 1
DOM_OFF = N_POSE
NONDOM_OFF = N_POSE + HAND_BLOCK
F = N_POSE + 2 * HAND_BLOCK
RAW_KEYS = ("hands", "hand_present", "hand_label", "pose", "pose_present", "width", "height")

ML = Path(__file__).resolve().parent


def load_raw(path):
    d = np.load(path)
    return {k: d[k] for k in RAW_KEYS}


def trim_range(hand_present):
    idx = np.flatnonzero(np.asarray(hand_present).any(axis=1))
    T = len(hand_present)
    if len(idx) == 0:
        return 0, T - 1
    return max(0, int(idx[0]) - PAD), min(T - 1, int(idx[-1]) + PAD)


def body_frame(pose, pose_present, a):
    m = np.asarray(pose_present).astype(bool)
    if not m.any():
        return a / 2, 0.5, FALLBACK_SCALE
    L, R = pose[m, 11], pose[m, 12]
    ox = np.median(((L[:, 0] + R[:, 0]) / 2) * a)
    oy = np.median((L[:, 1] + R[:, 1]) / 2)
    s = np.median(np.hypot((L[:, 0] - R[:, 0]) * a, L[:, 1] - R[:, 1]))
    return float(ox), float(oy), float(s) if s > 1e-6 else FALLBACK_SCALE


def assign_sides(hands, hand_present, hand_label, a, ox):
    T = len(hands)
    side = np.zeros((T, 2, 21, 3))
    present = np.zeros((T, 2), bool)
    for t in range(T):
        det = [k for k in (0, 1) if hand_present[t, k]]
        if len(det) == 1:
            k = det[0]
            lab = int(hand_label[t, k])
            s = lab if lab in (LEFT, RIGHT) else (RIGHT if hands[t, k, 0, 0] * a < ox else LEFT)
            side[t, s] = hands[t, k]
            present[t, s] = True
        elif len(det) == 2:
            l0, l1 = int(hand_label[t, 0]), int(hand_label[t, 1])
            if l0 != l1 and l0 in (LEFT, RIGHT) and l1 in (LEFT, RIGHT):
                s0 = l0
            else:
                s0 = RIGHT if hands[t, 0, 0, 0] <= hands[t, 1, 0, 0] else LEFT
            side[t, s0] = hands[t, 0]
            side[t, 1 - s0] = hands[t, 1]
            present[t] = True
    return side, present


def frame_features(raw):
    """Steps 3-8: raw sequence -> (N, F) float64 per-frame features + debug meta."""
    a = float(raw["width"]) / float(raw["height"])
    start, end = trim_range(raw["hand_present"])
    sl = slice(start, end + 1)
    hands = np.asarray(raw["hands"], np.float64)[sl]
    hp = np.asarray(raw["hand_present"])[sl]
    hl = np.asarray(raw["hand_label"])[sl]
    pose = np.asarray(raw["pose"], np.float64)[sl]
    pp = np.asarray(raw["pose_present"]).astype(bool)[sl]

    ox, oy, s = body_frame(pose, pp, a)
    side, present = assign_sides(hands, hp, hl, a, ox)
    dom = RIGHT if present[:, RIGHT].sum() >= present[:, LEFT].sum() else LEFT

    N = len(hands)
    out = np.zeros((N, F))
    out[:, 0:N_POSE:2] = (pose[:, POSE_IDX, 0] * a - ox) / s
    out[:, 1:N_POSE:2] = (pose[:, POSE_IDX, 1] - oy) / s
    out[~pp, :N_POSE] = 0
    for off, sd in ((DOM_OFF, dom), (NONDOM_OFF, 1 - dom)):
        h, m = side[:, sd], present[:, sd]
        blk = np.stack([(h[..., 0] * a - ox) / s,
                        (h[..., 1] - oy) / s,
                        (h[..., 2] - h[:, :1, 2]) * a / s], axis=-1).reshape(N, 63)
        blk[~m] = 0
        out[:, off:off + 63] = blk
        out[:, off + 63] = m
    meta = {"trim": [start, end], "origin": [ox, oy], "scale": s,
            "dominant": "right" if dom == RIGHT else "left"}
    return out, meta


def resample(seq, n=SEQ_LEN, pos=None):
    """Step 9. `pos` (fractional source indices) overrides uniform spacing; used for time-warp augmentation."""
    N = len(seq)
    if pos is None:
        pos = (np.arange(n) * (N - 1)) / (n - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, N - 1)
    i1 = np.minimum(i0 + 1, N - 1)
    w = (pos - i0)[:, None]
    return seq[i0] * (1 - w) + seq[i1] * w


def extract_features(raw):
    frames, meta = frame_features(raw)
    return resample(frames).astype(np.float32), meta


def attempt_stats(x):
    """Step 10, from a (32, F) feature array."""
    x = np.asarray(x, np.float64)
    st = {}
    for name, off in (("dom", DOM_OFF), ("nondom", NONDOM_OFF)):
        fl = x[:, off + 63]
        on = fl >= 0.5
        both = on[1:] & on[:-1]
        d = np.hypot(np.diff(x[:, off]), np.diff(x[:, off + 1]))
        st[f"{name}_presence"] = float(fl.mean())
        st[f"{name}_wrist_y"] = float(x[on, off + 1].mean()) if on.any() else None
        st[f"{name}_movement"] = float(d[both].sum())
    st["hands_used"] = int(st["dom_presence"] >= 0.3) + int(st["nondom_presence"] >= 0.3)
    pose_on = np.any(x[:, :N_POSE] != 0, axis=1)
    st["nose_y"] = float(x[pose_on, 1].mean()) if pose_on.any() else None
    return st


def _golden_case(name, raw, note):
    raw = {k: (np.round(np.asarray(v, np.float64), 6) if k in ("hands", "pose") else np.asarray(v))
           for k, v in raw.items()}
    feats, meta = extract_features(raw)
    return {
        "name": name,
        "note": note,
        "input": {
            "width": int(raw["width"]), "height": int(raw["height"]),
            "hands": raw["hands"].tolist(),
            "hand_present": raw["hand_present"].astype(int).tolist(),
            "hand_label": raw["hand_label"].astype(int).tolist(),
            "pose": raw["pose"].tolist(),
            "pose_present": raw["pose_present"].astype(int).tolist(),
        },
        "expected": {**meta, "features": feats.astype(float).tolist(), "stats": attempt_stats(feats)},
    }


def _window(raw, margin=15):
    idx = np.flatnonzero(raw["hand_present"].any(axis=1))
    sl = slice(max(0, idx[0] - margin), idx[-1] + margin + 1)
    return {k: (v[sl] if np.ndim(v) else v) for k, v in raw.items()}


def make_golden():
    lm = ML / "landmarks"
    # Case 1: real two-handed clip (HOW ARE YOU) with same-label frames -> exercises x-based resolution.
    c1 = _window(load_raw(lm / "4_2.npz"))
    # Case 2: real one-handed clip (RED) with injected edge cases: unknown handedness labels,
    # missing-pose frames, and a non-16:9 size to exercise aspect correction.
    c2 = _window(load_raw(lm / "74_6.npz"))
    c2 = {k: (np.array(v, copy=True) if np.ndim(v) else v) for k, v in c2.items()}
    on = np.flatnonzero(c2["hand_present"].any(axis=1))
    c2["hand_label"][on[3], :] = -1
    c2["hand_label"][on[10], :] = -1
    c2["pose_present"][on[5]] = 0
    c2["pose"][on[5]] = 0
    c2["pose_present"][on[6]] = 0
    c2["pose"][on[6]] = 0
    c2["width"], c2["height"] = np.int32(640), np.int32(480)
    cases = [
        _golden_case("two_hands_same_label", c1,
                     "clip 4_2 (HOW ARE YOU), windowed; contains frames where both hands share a label"),
        _golden_case("one_hand_edge_cases", c2,
                     "clip 74_6 (RED), windowed; hand_label=-1 injected on 2 hand frames, pose removed on "
                     "2 frames, width/height overridden to 640x480"),
    ]
    out = ML / "fixtures" / "feature_golden.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({"spec_version": SPEC_VERSION, "tolerance": 1e-5, "feature_dim": F, "seq_len": SEQ_LEN,
               "cases": cases}, open(out, "w"))

    # Self-check: features recomputed from the JSON-serialized inputs must match.
    for case in json.load(open(out))["cases"]:
        raw = {k: np.array(v) for k, v in case["input"].items()}
        got, _ = extract_features(raw)
        diff = np.abs(got - np.array(case["expected"]["features"])).max()
        assert diff == 0, f"{case['name']}: roundtrip diff {diff}"
        e = case["expected"]
        print(f"{case['name']}: T={len(raw['hands'])} trim={e['trim']} dominant={e['dominant']} "
              f"scale={e['scale']:.4f} -> features {got.shape}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-golden", action="store_true")
    if ap.parse_args().make_golden:
        make_golden()
