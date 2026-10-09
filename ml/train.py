"""Train the Kumpas sign classifier (2-layer GRU) on extracted FSL-105 landmarks.

Uses the dataset's train/test split. 15% of the train split (stratified, fixed
seed) is held out as validation for checkpoint selection; the test split is
only used for the final report.

  python train.py            # trains, saves ml/checkpoints/best.pt, writes ml/report.md
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import features as feat

ML = Path(__file__).resolve().parent
LANDMARKS = ML / "landmarks"
CKPT = ML / "checkpoints" / "best.pt"
REPORT = ML / "report.md"
CM_PNG = ML / "confusion_matrix.png"
ONNX_MARK = ("<!-- onnx-parity:start -->", "<!-- onnx-parity:end -->")

HIDDEN, LAYERS, DROPOUT = 128, 2, 0.3

# Coordinate index groups for spatial augmentation: (x idx, y idx, z idx or None, presence flag idx or None)
_P = np.arange(len(feat.POSE_IDX))
GROUPS = [(2 * _P, 2 * _P + 1, None, None)] + [
    (off + 3 * np.arange(21), off + 3 * np.arange(21) + 1, off + 3 * np.arange(21) + 2, off + 63)
    for off in (feat.DOM_OFF, feat.NONDOM_OFF)
]


class SignGRU(nn.Module):
    def __init__(self, n_feat, n_classes, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.gru = nn.GRU(n_feat, HIDDEN, num_layers=LAYERS, batch_first=True, dropout=DROPOUT)
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN, n_classes)

    def forward(self, x):
        out, _ = self.gru((x - self.mean) / self.std)
        return self.fc(self.drop(out[:, -1]))


def load_dataset():
    classes = json.load(open(ML / "classes.json"))
    by_label = {c["dataset_label_id"]: c["id"] for c in classes}
    items = []
    for p in sorted(LANDMARKS.glob("*.npz")):
        d = np.load(p)
        if int(d["label_id"]) not in by_label:
            continue
        frames, meta = feat.frame_features({k: d[k] for k in feat.RAW_KEYS})
        items.append({"clip_id": p.stem, "y": by_label[int(d["label_id"])], "split": str(d["split"]),
                      "label_id": int(d["label_id"]), "clip_num": int(p.stem.split("_")[1]),
                      "body_sig": body_signature(d),
                      "frames": frames, "meta": meta,
                      "x": feat.resample(frames).astype(np.float32)})
    assign_signers(items)
    return classes, items


# Visually verified per-class signer run lengths where the automatic segmentation is wrong.
SIGNER_RUNS_OVERRIDE = {89: [5, 5, 5, 5]}


def body_signature(d):
    """Signer-specific body geometry from rest frames (median pose): used only to identify signers."""
    P = d["pose"][:, :, :2].astype(np.float64).copy()
    P[..., 0] *= float(d["width"]) / float(d["height"])
    rest = ~d["hand_present"].any(1)
    m = np.median(P[rest] if rest.sum() > 5 else P, 0)
    sw = np.linalg.norm(m[11] - m[12])
    neck = (m[11, 1] + m[12, 1]) / 2 - m[0, 1]
    return np.array([sw, np.linalg.norm(m[2] - m[5]) / sw, np.linalg.norm(m[9] - m[10]) / sw,
                     neck, neck / sw, np.linalg.norm(m[7] - m[8]) / sw])


def segment_runs(Z, k=4, min_len=3):
    """Split an ordered sequence into k contiguous runs minimizing within-run variance (DP)."""
    n = len(Z)
    cs = np.vstack([np.zeros(Z.shape[1]), np.cumsum(Z, 0)])
    cs2 = np.concatenate([[0], np.cumsum((Z ** 2).sum(1))])
    cost = lambda i, j: cs2[j] - cs2[i] - ((cs[j] - cs[i]) ** 2).sum() / (j - i)
    D = np.full((k + 1, n + 1), np.inf)
    B = np.zeros((k + 1, n + 1), int)
    D[0, 0] = 0
    for r in range(1, k + 1):
        for j in range(r * min_len, n + 1):
            for i in range((r - 1) * min_len, j - min_len + 1):
                v = D[r - 1, i] + cost(i, j)
                if v < D[r, j]:
                    D[r, j], B[r, j] = v, i
    bounds = [n]
    for r in range(k, 0, -1):
        bounds.append(B[r, bounds[-1]])
    bounds = bounds[::-1]
    return [bounds[i + 1] - bounds[i] for i in range(k)]


def assign_signers(items):
    """FSL-105 has 4 signers who appear in the same order (A, B, C, D) as contiguous runs of clip numbers
    in every class, but run lengths vary (5-6), so clip_num // 5 is wrong. Segment each class's clips
    by body signature into 4 ordered runs. Verified visually on all 24 chosen classes."""
    by_label = {}
    for it in items:
        by_label.setdefault(it["label_id"], []).append(it)
    for lid, its in by_label.items():
        its.sort(key=lambda it: it["clip_num"])
        X = np.stack([it["body_sig"] for it in its])
        runs = SIGNER_RUNS_OVERRIDE.get(lid) or segment_runs((X - X.mean(0)) / (X.std(0) + 1e-9))
        signer = np.repeat(np.arange(len(runs)), runs)
        for it, sg in zip(its, signer):
            it["signer"] = int(sg)


def stratified_val(pool, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    train, val, by_class = [], [], {}
    for it in pool:
        by_class.setdefault(it["y"], []).append(it)
    for _, lst in sorted(by_class.items()):
        idx = rng.permutation(len(lst))
        n_val = max(1, round(len(lst) * val_frac))
        val += [lst[i] for i in idx[:n_val]]
        train += [lst[i] for i in idx[n_val:]]
    return train, val


def split_items(items, val_frac=0.15, seed=0):
    """Dataset train/test split; validation carved from the train split."""
    train, val = stratified_val([it for it in items if it["split"] == "train"], val_frac, seed)
    return train, val, [it for it in items if it["split"] == "test"]


def augment(frames, rng):
    """Temporal crop, frame dropout, rotation/scale/translation jitter, noise, time warp. Never mirrors."""
    x = frames.copy()
    N = len(x)
    c0, c1 = rng.integers(0, int(0.1 * N) + 1, size=2)
    if N - c0 - c1 >= 8:
        x = x[c0:N - c1]
    keep = rng.random(len(x)) >= rng.uniform(0, 0.15)
    if keep.sum() >= 8:
        x = x[keep]

    th = np.deg2rad(rng.uniform(-10, 10))
    sc = rng.uniform(0.9, 1.1)
    tx, ty = rng.uniform(-0.1, 0.1, size=2)
    c, s = np.cos(th), np.sin(th)
    pose_on = np.any(x[:, :feat.N_POSE] != 0, axis=1, keepdims=True)
    for xi, yi, zi, fi in GROUPS:
        m = pose_on if fi is None else (x[:, [fi]] > 0.5)
        X, Y = x[:, xi], x[:, yi]
        x[:, xi] = (sc * (c * X - s * Y) + tx + rng.normal(0, 0.01, X.shape)) * m
        x[:, yi] = (sc * (s * X + c * Y) + ty + rng.normal(0, 0.01, Y.shape)) * m
        if zi is not None:
            x[:, zi] *= sc

    g = np.exp(rng.uniform(-0.3, 0.3))
    pos = np.linspace(0, 1, feat.SEQ_LEN) ** g * (len(x) - 1)
    return feat.resample(x, pos=pos).astype(np.float32)


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    return torch.softmax(model(torch.as_tensor(X, device=device)), -1).cpu().numpy()


def confusion(probs, y, C):
    pred = probs.argmax(1)
    cm = np.zeros((C, C), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    rates = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    confused = sorted(((rates[i, j], i, j) for i in range(C) for j in range(C)
                       if i != j and rates[i, j] > 0.15), reverse=True)
    return cm, rates, confused


def write_report(classes, probs, y, info, loso=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = len(classes)
    names = [c["gloss"] for c in classes]
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, 1)[:, :3]
    top1_acc = (pred == y).mean()
    top3_acc = np.mean([y[i] in top3[i] for i in range(len(y))])
    cm, rates, confused = confusion(probs, y, C)
    fig, ax = plt.subplots(figsize=(12, 10.5))
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C), names, rotation=90, fontsize=8)
    ax.set_yticks(range(C), names, fontsize=8)
    for i in range(C):
        for j in range(C):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if rates[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Kumpas test confusion matrix (counts; color = row rate)  top-1 {top1_acc:.1%}")
    fig.tight_layout()
    fig.savefig(CM_PNG, dpi=110)
    plt.close(fig)

    n_per = cm.sum(1)
    lines = [
        "# Kumpas classifier report",
        "",
        "Summary, confusions and per-class numbers are measured on the FSL-105 **test split** for the 24 chosen classes "
        "(generated by `ml/train.py`).",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Top-1 accuracy | **{top1_acc:.1%}** ({(pred == y).sum()}/{len(y)}) |",
        f"| Top-3 accuracy | **{top3_acc:.1%}** |",
        f"| Classes | {C} |",
        f"| Clips: train / val / test | {info['n_train']} / {info['n_val']} / {len(y)} |",
        f"| Best val accuracy (checkpoint selection) | {info['best_val_acc']:.1%} at epoch {info['best_epoch']} |",
        f"| Model | 2-layer GRU, hidden {HIDDEN}, dropout {DROPOUT}, {info['params']:,} params |",
        f"| Input | [1, {feat.SEQ_LEN}, {feat.F}] (feature spec v{feat.SPEC_VERSION}, see FEATURES.md) |",
        f"| Training | {info['epochs']} epochs on `{info['device']}`, {info['train_secs']:.0f} s, seed {info['seed']} |",
        "",
        f"Caveat: the test split has only {n_per.min()}–{n_per.max()} clips per class, so a single "
        f"error moves a class's accuracy by {1 / n_per.max():.0%}–{1 / n_per.min():.0%}. "
        "Per-class numbers and confusions are noisy.",
        "",
        "**Signer overlap:** FSL-105 has 4 signers, and the official split puts clips from all 4 in both "
        "train and test. So this split measures accuracy on *signers seen in training*. See "
        "\"Unseen-signer accuracy\" below for the number that better predicts a new user.",
        "",
        "## Confused pairs (> 15% of a class's test clips)",
        "",
    ]
    if confused:
        lines += ["| True | Predicted as | Rate | Count |", "|---|---|---|---|"]
        lines += [f"| {names[i]} | {names[j]} | {r:.0%} | {cm[i, j]}/{n_per[i]} |" for r, i, j in confused]
    else:
        lines.append("None.")
    lines += ["", "## Per-class accuracy", "", "| Class | Accuracy | Correct / clips |", "|---|---|---|"]
    for i in np.argsort([cm[i, i] / max(n_per[i], 1) for i in range(C)]):
        lines.append(f"| {names[i]} | {cm[i, i] / max(n_per[i], 1):.0%} | {cm[i, i]}/{n_per[i]} |")
    lines += ["", "## Confusion matrix", "", f"![confusion matrix]({CM_PNG.name})", ""]
    if loso:
        folds, lp, ly = loso
        lcm, lrates, lconf = confusion(lp, ly, C)
        l1 = (lp.argmax(1) == ly).mean()
        l3 = np.mean([ly[i] in np.argsort(-lp[i])[:3] for i in range(len(ly))])
        lines += [
            "## Unseen-signer accuracy (leave-one-signer-out)", "",
            "FSL-105 has 4 signers; each clip's signer was identified by body-geometry segmentation and "
            "verified visually on all 24 classes (`assign_signers`). Each fold trains on 3 "
            "signers (15% held out for checkpoint selection) and tests on every clip of the 4th signer. "
            "Same model, features and training settings as above.", "",
            "| Held-out signer | Clips | Top-1 | Top-3 |", "|---|---|---|---|",
            *[f"| {f['signer']} | {f['n']} | {f['top1']:.1%} | {f['top3']:.1%} |" for f in folds],
            f"| **All folds pooled** | {len(ly)} | **{l1:.1%}** | **{l3:.1%}** |", "",
            f"Mean of folds: top-1 {np.mean([f['top1'] for f in folds]):.1%}, "
            f"top-3 {np.mean([f['top3'] for f in folds]):.1%}.", "",
            "### Unseen-signer confused pairs (> 15%, pooled over all folds)", "",
            "Based on ~20 clips per class, so less noisy than the test-split list above.", "",
        ]
        if lconf:
            ln = lcm.sum(1)
            lines += ["| True | Predicted as | Rate | Count |", "|---|---|---|---|"]
            lines += [f"| {names[i]} | {names[j]} | {r:.0%} | {lcm[i, j]}/{ln[i]} |" for r, i, j in lconf]
        else:
            lines.append("None.")
        lines += ["", "### Unseen-signer per-class accuracy (pooled)", "", "| Class | Accuracy |", "|---|---|"]
        lines += [f"| {names[i]} | {lrates[i, i]:.0%} ({lcm[i, i]}/{lcm[i].sum()}) |"
                  for i in np.argsort(np.diag(lrates))]
        lines.append("")
    lines += [
              ONNX_MARK[0], "_ONNX parity not checked yet (run `ml/export.py`)._", ONNX_MARK[1], ""]
    REPORT.write_text("\n".join(lines))
    return top1_acc, top3_acc, confused


def fit(train, val, n_classes, args, device, ckpt_path, tag=""):
    """Train with augmentation; keep the checkpoint with best val accuracy (ties -> lower val loss)."""
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    Xtr_uniform = np.stack([it["x"] for it in train]).reshape(-1, feat.F)
    mean, std = Xtr_uniform.mean(0), np.maximum(Xtr_uniform.std(0), 1e-3)
    Xva, yva = np.stack([it["x"] for it in val]), np.array([it["y"] for it in val])
    ytr = torch.tensor([it["y"] for it in train], device=device)

    model = SignGRU(feat.F, n_classes, mean, std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_epoch = (-1.0, float("inf")), 0
    ckpt_path.parent.mkdir(exist_ok=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            X = torch.as_tensor(np.stack([augment(train[i]["frames"], rng) for i in idx]), device=device)
            loss = loss_fn(model(X), ytr[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        pv = predict(model, Xva, device)
        val_acc = (pv.argmax(1) == yva).mean()
        val_loss = float(-np.log(pv[np.arange(len(yva)), yva] + 1e-9).mean())
        if (val_acc, -val_loss) > (best[0], -best[1]):
            best, best_epoch = (val_acc, val_loss), epoch
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                        "n_feat": feat.F, "spec_version": feat.SPEC_VERSION, "epoch": epoch,
                        "val_acc": float(val_acc)}, ckpt_path)
        if epoch % 20 == 0 or epoch == 1:
            print(f"{tag}epoch {epoch:3d} | train loss {total / len(train):.3f} | val loss {val_loss:.3f} "
                  f"acc {val_acc:.1%} | best {best[0]:.1%} @ {best_epoch} | {time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"])
    return model, {"best_val_acc": best[0], "best_epoch": best_epoch, "train_secs": time.time() - t0,
                   "params": sum(p.numel() for p in model.parameters())}


def run_loso(classes, items, args, device):
    """Leave-one-signer-out: train on 3 signers, test on the unseen 4th (both dataset splits pooled)."""
    folds, all_probs, all_y = [], [], []
    for s in sorted({it["signer"] for it in items}):
        test = [it for it in items if it["signer"] == s]
        train, val = stratified_val([it for it in items if it["signer"] != s], seed=args.seed)
        model, info = fit(train, val, len(classes), args, device,
                          ML / "checkpoints" / f"loso_signer{s}.pt", tag=f"[LOSO signer {s}] ")
        probs = predict(model, np.stack([it["x"] for it in test]), device)
        y = np.array([it["y"] for it in test])
        top3 = np.argsort(-probs, 1)[:, :3]
        folds.append({"signer": s, "n": len(y), "top1": float((probs.argmax(1) == y).mean()),
                      "top3": float(np.mean([y[i] in top3[i] for i in range(len(y))]))})
        print(f"[LOSO signer {s}] unseen-signer top-1 {folds[-1]['top1']:.1%} top-3 {folds[-1]['top3']:.1%}")
        all_probs.append(probs)
        all_y.append(y)
    return folds, np.concatenate(all_probs), np.concatenate(all_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-loso", action="store_true", help="skip the leave-one-signer-out evaluation")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    classes, items = load_dataset()
    train, val, test = split_items(items, seed=args.seed)
    print(f"{len(items)} clips: train {len(train)}, val {len(val)}, test {len(test)} | device {device}")

    model, info = fit(train, val, len(classes), args, device, CKPT)
    ck = torch.load(CKPT, weights_only=False)
    ck["classes"] = classes
    torch.save(ck, CKPT)
    info.update(n_train=len(train), n_val=len(val), epochs=args.epochs, device=str(device), seed=args.seed)
    test_probs = predict(model, np.stack([it["x"] for it in test]), device)
    loso = None if args.no_loso else run_loso(classes, items, args, device)

    top1, top3, confused = write_report(classes, test_probs, np.array([it["y"] for it in test]), info, loso)
    print(f"\nTEST top-1 {top1:.1%} | top-3 {top3:.1%} | checkpoint epoch {info['best_epoch']} "
          f"(val {info['best_val_acc']:.1%})")
    for r, i, j in confused:
        print(f"  {classes[i]['gloss']} -> {classes[j]['gloss']}: {r:.0%}")
    print(f"wrote {CKPT}, {REPORT}, {CM_PNG}")

if __name__ == "__main__":
    main()
