"""Extract raw per-frame MediaPipe landmarks from every FSL-105 clip.

Uses the MediaPipe Tasks API (HandLandmarker + PoseLandmarker lite) in VIDEO
mode with the same .task files the web app serves from web/public/mediapipe/.
No normalization is applied here; see features.py for that.

Output: ml/landmarks/<clip_id>.npz where clip_id = "<label_id>_<clip_num>".
  hands          (T, 2, 21, 3) float32  normalized image x, y and MediaPipe z;
                                        slots are in MediaPipe detection order
                                        (NOT remapped to left/right), zeros if absent
  hand_present   (T, 2) uint8           1 if a hand was detected in that slot
  hand_label     (T, 2) int8            MediaPipe handedness: 0=Left, 1=Right, -1=absent
                                        (raw label on the unmirrored video frame)
  hand_score     (T, 2) float32         handedness confidence
  pose           (T, 33, 4) float32     x, y, z, visibility for all 33 pose landmarks
  pose_present   (T,) uint8
  timestamps_ms  (T,) int64             timestamps fed to detect_for_video
  label_id, split, fps, width, height, src (scalars / strings)

Usage:
  python extract.py               # all clips, resumable
  python extract.py --limit 5     # quick test
"""

import argparse
import csv
import multiprocessing as mp_proc
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "fsl105"
MODELS = ROOT / "web" / "public" / "mediapipe"
HAND_MODEL = MODELS / "hand_landmarker.task"
POSE_MODEL = MODELS / "pose_landmarker_lite.task"
OUT = ROOT / "ml" / "landmarks"
HAND_LABELS = {"Left": 0, "Right": 1}


def load_clips():
    clips = []
    for split in ("train", "test"):
        with open(DATASET / f"{split}.csv", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rel = Path(*row["vid_path"].replace("\\", "/").split("/"))
                label_id = int(row["id_label"])
                clips.append({
                    "clip_id": f"{label_id}_{rel.stem}",
                    "path": str(DATASET / rel),
                    "src": rel.as_posix(),
                    "label_id": label_id,
                    "split": split,
                })
    return clips


def make_landmarkers():
    from mediapipe.tasks.python import BaseOptions, vision

    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    ))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    ))
    return hand, pose


def extract_clip(clip):
    import mediapipe as mp

    cap = cv2.VideoCapture(clip["path"])
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hands, hand_present, hand_label, hand_score = [], [], [], []
    pose, pose_present, timestamps = [], [], []
    # Fresh landmarkers per clip so VIDEO-mode tracking never carries over between clips.
    hand_lm, pose_lm = make_landmarkers()
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = int(round(i * 1000.0 / fps))
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            hr = hand_lm.detect_for_video(image, ts)
            pr = pose_lm.detect_for_video(image, ts)

            h = np.zeros((2, 21, 3), np.float32)
            hp = np.zeros(2, np.uint8)
            hl = np.full(2, -1, np.int8)
            hs = np.zeros(2, np.float32)
            for k, (lms, cats) in enumerate(zip(hr.hand_landmarks[:2], hr.handedness[:2])):
                h[k] = [(p.x, p.y, p.z) for p in lms]
                hp[k] = 1
                hl[k] = HAND_LABELS.get(cats[0].category_name, -1)
                hs[k] = cats[0].score

            p = np.zeros((33, 4), np.float32)
            if pr.pose_landmarks:
                p[:] = [(q.x, q.y, q.z, q.visibility or 0.0) for q in pr.pose_landmarks[0]]

            hands.append(h); hand_present.append(hp); hand_label.append(hl); hand_score.append(hs)
            pose.append(p); pose_present.append(1 if pr.pose_landmarks else 0); timestamps.append(ts)
            i += 1
    finally:
        hand_lm.close()
        pose_lm.close()
        cap.release()

    if not hands:
        raise RuntimeError("no frames decoded")
    return dict(
        hands=np.stack(hands), hand_present=np.stack(hand_present),
        hand_label=np.stack(hand_label), hand_score=np.stack(hand_score),
        pose=np.stack(pose), pose_present=np.array(pose_present, np.uint8),
        timestamps_ms=np.array(timestamps, np.int64),
        label_id=np.int64(clip["label_id"]), split=np.str_(clip["split"]),
        fps=np.float32(fps), width=np.int32(width), height=np.int32(height),
        src=np.str_(clip["src"]),
    )


def worker(clip):
    out = OUT / f"{clip['clip_id']}.npz"
    tmp = OUT / f".{clip['clip_id']}.partial"
    t0 = time.time()
    try:
        data = extract_clip(clip)
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(tmp, out)
        return clip["clip_id"], None, time.time() - t0, len(data["timestamps_ms"])
    except Exception:
        tmp.unlink(missing_ok=True)
        return clip["clip_id"], traceback.format_exc(), time.time() - t0, 0


def init_worker():
    cv2.setNumThreads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process the first N pending clips")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    for m in (HAND_MODEL, POSE_MODEL):
        if not m.exists():
            sys.exit(f"missing model: {m}")
    OUT.mkdir(parents=True, exist_ok=True)

    clips = load_clips()
    pending = [c for c in clips if not (OUT / f"{c['clip_id']}.npz").exists()]
    done = len(clips) - len(pending)
    if args.limit:
        pending = pending[:args.limit]
    print(f"{len(clips)} clips total, {done} already done, "
          f"processing {len(pending)} with {args.workers} workers", flush=True)
    if not pending:
        return

    fail_log = OUT / "_failures.log"
    failures = 0
    t0 = time.time()
    with mp_proc.get_context("spawn").Pool(args.workers, initializer=init_worker) as pool:
        for n, (clip_id, err, dt, frames) in enumerate(pool.imap_unordered(worker, pending), 1):
            if err:
                failures += 1
                with open(fail_log, "a") as f:
                    f.write(f"=== {clip_id} ({time.strftime('%Y-%m-%d %H:%M:%S')})\n{err}\n")
            elapsed = time.time() - t0
            eta = elapsed / n * (len(pending) - n)
            if err or n % 10 == 0 or n == len(pending):
                print(f"[{n}/{len(pending)}] {clip_id} {'FAILED' if err else f'{frames}f {dt:.1f}s'} "
                      f"| elapsed {elapsed / 60:.1f}m | ETA {eta / 60:.1f}m | failures {failures}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f}m, {failures} failures"
          + (f" (see {fail_log})" if failures else ""), flush=True)


if __name__ == "__main__":
    main()
