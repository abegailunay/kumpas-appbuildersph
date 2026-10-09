"""Export the best checkpoint to ONNX + web assets, then verify ONNX Runtime matches PyTorch.

Writes to web/public/models/:
  kumpas.onnx          input "features" [1, 32, F] float32 -> output "probs" [1, C] (softmax);
                       per-feature standardization is baked into the graph
  classes.json         copy of ml/classes.json
  norm.json            mean/std (already applied inside the model) + I/O metadata
  class_profiles.json  per-class attempt-stat profiles (FEATURES.md section 10) for Teach tips
"""

import json
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

import features as feat
from train import CKPT, ML, ONNX_MARK, REPORT, SignGRU, load_dataset, split_items

OUT = ML.parent / "web" / "public" / "models"
STAT_KEYS = ("dom_presence", "nondom_presence", "dom_wrist_y", "nondom_wrist_y",
             "dom_movement", "nondom_movement", "nose_y")


class WithSoftmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        return torch.softmax(self.model(features), dim=-1)


def class_profiles(classes, items):
    profiles = {}
    for c in classes:
        its = [it for it in items if it["y"] == c["id"]]
        stats = [feat.attempt_stats(it["x"]) for it in its]
        p = {"gloss": c["gloss"], "n_clips": len(its),
             "hands_used_mean": float(np.mean([s["hands_used"] for s in stats])),
             "dominant_right_frac": float(np.mean([it["meta"]["dominant"] == "right" for it in its]))}
        for k in STAT_KEYS:
            v = [s[k] for s in stats if s[k] is not None]
            p[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} if v else None
        profiles[str(c["id"])] = p
    return {"spec_version": feat.SPEC_VERSION,
            "source": "FSL-105 train split clips, unaugmented; stats per FEATURES.md section 10",
            "units": "wrist_y / nose_y / movement in shoulder widths; positive y is below the shoulder line",
            "classes": profiles}


def main():
    ck = torch.load(CKPT, weights_only=False)
    classes = ck["classes"]
    model = SignGRU(ck["n_feat"], len(classes), ck["mean"], ck["std"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    wrapped = WithSoftmax(model).eval()

    OUT.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT / "kumpas.onnx"
    torch.onnx.export(wrapped, (torch.zeros(1, feat.SEQ_LEN, ck["n_feat"]),), str(onnx_path),
                      opset_version=17, input_names=["features"], output_names=["probs"], dynamo=False)
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    shutil.copy(ML / "classes.json", OUT / "classes.json")
    json.dump({"spec_version": feat.SPEC_VERSION, "seq_len": feat.SEQ_LEN, "feature_dim": ck["n_feat"],
               "input": "features", "output": "probs", "output_is_softmax": True,
               "standardization_applied_in_model": True,
               "note": "mean/std are already applied inside kumpas.onnx; do NOT apply them again in the browser",
               "mean": np.asarray(ck["mean"]).tolist(), "std": np.asarray(ck["std"]).tolist()},
              open(OUT / "norm.json", "w"))

    _, items = load_dataset()
    train, val, test = split_items(items)
    json.dump(class_profiles(classes, train + val), open(OUT / "class_profiles.json", "w"), indent=1)

    # Parity check on the full test split, batch size 1 as in the browser.
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y = np.array([it["y"] for it in test])
    p_ort, p_pt = [], []
    with torch.no_grad():
        for it in test:
            x = it["x"][None]
            p_ort.append(sess.run(["probs"], {"features": x})[0][0])
            p_pt.append(wrapped(torch.from_numpy(x))[0].numpy())
    p_ort, p_pt = np.array(p_ort), np.array(p_pt)
    max_diff = float(np.abs(p_ort - p_pt).max())
    agree = float((p_ort.argmax(1) == p_pt.argmax(1)).mean())
    acc_ort, acc_pt = float((p_ort.argmax(1) == y).mean()), float((p_pt.argmax(1) == y).mean())

    section = "\n".join([
        ONNX_MARK[0], "## ONNX parity (`ml/export.py`)", "",
        f"`web/public/models/kumpas.onnx` (opset 17, {onnx_path.stat().st_size / 1e3:.0f} KB) vs PyTorch "
        f"(CPU) on all {len(test)} test clips, batch size 1:", "",
        "| Check | Value |", "|---|---|",
        f"| Max abs prob difference | {max_diff:.2e} |",
        f"| Top-1 agreement | {agree:.1%} |",
        f"| Top-1 accuracy, ONNX Runtime | {acc_ort:.1%} |",
        f"| Top-1 accuracy, PyTorch | {acc_pt:.1%} |",
        ONNX_MARK[1]])
    text = REPORT.read_text()
    a, b = text.index(ONNX_MARK[0]), text.index(ONNX_MARK[1]) + len(ONNX_MARK[1])
    REPORT.write_text(text[:a] + section + text[b:])

    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e3:.0f} KB) + classes.json, norm.json, class_profiles.json")
    print(f"parity on {len(test)} test clips: max |dprob| {max_diff:.2e}, argmax agreement {agree:.1%}, "
          f"acc ORT {acc_ort:.1%} vs PyTorch {acc_pt:.1%}")
    assert max_diff < 1e-4 and agree == 1.0, "ONNX output does not match PyTorch"


if __name__ == "__main__":
    main()
