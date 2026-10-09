"""Make one small reference clip per class from FSL-105 into web/public/clips/.

Uses the extracted landmarks to trim each clip to the signing part (first..last frame with a hand,
plus a short margin), then encodes 640x360 VP9 WebM without audio via ffmpeg.

  python make_clips.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parent
DATASET = ML.parent / "datasets" / "fsl105"
OUT = ML.parent / "web" / "public" / "clips"
MARGIN_S = 0.25
CLIP_NUM = 0  # clip 0 of every class is the same signer, so all reference videos look consistent
ATTRIBUTION = ("FSL-105 dataset by Isaiah Jassen Tupal and Melvin Cabatuan, De La Salle University, "
               "CC BY 4.0, https://doi.org/10.17632/48y2y99mb9.2. Trimmed and re-encoded.")


def slug(s):
    return "".join(c if c.isalnum() else "-" for c in s.lower()).strip("-").replace("--", "-")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in json.load(open(ML / "classes.json")):
        lid = c["dataset_label_id"]
        d = np.load(ML / "landmarks" / f"{lid}_{CLIP_NUM}.npz")
        fps, T = float(d["fps"]), len(d["hand_present"])
        idx = np.flatnonzero(d["hand_present"].any(1))
        start = max(0.0, idx[0] / fps - MARGIN_S)
        end = min(T / fps, (idx[-1] + 1) / fps + MARGIN_S)
        name = f"{c['id']:02d}-{slug(c['gloss'])}.webm"
        src = f"clips/{lid}/{CLIP_NUM}.MOV"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(DATASET / src),
            "-an", "-vf", "scale=640:360,fps=30", "-c:v", "libvpx-vp9", "-crf", "40", "-b:v", "0",
            "-row-mt", "1", "-deadline", "good", "-cpu-used", "2", str(OUT / name),
        ], check=True)
        size_kb = (OUT / name).stat().st_size / 1e3
        manifest.append({"class_id": c["id"], "gloss": c["gloss"], "file": name, "source": src,
                         "start_s": round(start, 3), "duration_s": round(end - start, 3)})
        print(f"{name:32s} {start:5.2f}-{end:5.2f}s  {size_kb:6.0f} KB")
    json.dump({"attribution": ATTRIBUTION, "clips": manifest}, open(OUT / "manifest.json", "w"), indent=1)
    total = sum((OUT / m["file"]).stat().st_size for m in manifest) / 1e6
    print(f"{len(manifest)} clips, {total:.1f} MB total -> {OUT}")


if __name__ == "__main__":
    main()
