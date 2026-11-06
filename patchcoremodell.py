#!/usr/bin/env python3
"""
PatchCore-Baseline (Experiment E2) auf den mit preprocess_bottles.py erzeugten Sequenzen.

Ablauf
  1. Training: nur Gutflaschen der Trainings-Folds, Frame-Modus. Pro Phasenbin wird eine
     Memory Bank aus Patch-Features aufgebaut und per Greedy-Coreset ausgedünnt.
  2. Auswertung: jeder Frame wird mit der Bank seiner Phase verglichen
     (Nearest-Neighbor-Distanz je Patch -> Anomaliekarte -> Frame-Score = Maximum).
  3. Flaschen-Score = Maximum bzw. Mittel der Top-k Frame-Scores über alle Frames der Flasche.

  --phase-bins 1   entspricht dem ursprünglichen, positionsunabhängigen PatchCore
  --phase-bins 10  jede Bank sieht nur Frames aus demselben Zehntel der Durchfahrt

Beispiel
  python patchcore_baseline.py --inputs raw/inputs --out runs/pc_bins10 \
      --train-folds 0 1 2 3 --eval-folds 4 --phase-bins 10 --coreset-ratio 0.01
  Testset (erst ganz am Ende): --eval-split test  (ohne --eval-folds)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from bottle_dataset import BottleDataset

IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# Eingabe und Features
# --------------------------------------------------------------------------- #
def to_imagenet(x, mean, std):
    """Dataset-Normierung rückgängig machen, auf 3 Kanäle erweitern, ImageNet-normieren."""
    x = (x * std + mean).clamp(0, 1).repeat(1, 3, 1, 1)
    return (x - IMN_MEAN.to(x.device)) / IMN_STD.to(x.device)


class FeatureExtractor(torch.nn.Module):
    """WideResNet50, layer2 + layer3, 3x3-Nachbarschaftsmittel, Reduktion auf out_dim."""

    def __init__(self, out_dim=1024, random_weights=False):
        super().__init__()
        w = None if random_weights else Wide_ResNet50_2_Weights.IMAGENET1K_V1
        m = wide_resnet50_2(weights=w)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2, self.l3 = m.layer2, m.layer3
        self.out_dim = out_dim
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):                                   # x: (B, 3, H, W)
        f2 = self.l2(self.stem(x))                          # Stride 8
        f3 = self.l3(f2)                                    # Stride 16
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)                          # (B, 1536, h, w)
        B, C, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(B * h * w, C)
        f = F.adaptive_avg_pool1d(f.unsqueeze(1), self.out_dim).squeeze(1)
        return f.reshape(B, h, w, self.out_dim), (h, w)


# --------------------------------------------------------------------------- #
# Phasenbins
# --------------------------------------------------------------------------- #
def phase_bin(phase, n_bins):
    return np.minimum((np.asarray(phase) * n_bins).astype(int), n_bins - 1)


def item_bins(ds, n_bins):
    nf = ds.df.n_frames.to_numpy()
    ph = np.array([t / nf[i] for i, t in ds.items])
    return phase_bin(ph, n_bins)


# --------------------------------------------------------------------------- #
# Memory Bank
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect(ds_sub, ds, extractor, args, dev):
    out = []
    for batch in DataLoader(ds_sub, batch_size=args.bs, num_workers=args.workers):
        f, _ = extractor(to_imagenet(batch["x"].to(dev), ds.mean, ds.std))
        f = f.reshape(len(f), -1, f.shape[-1])
        for p in f:
            if args.patches_per_frame and args.patches_per_frame < len(p):
                p = p[torch.randperm(len(p), device=p.device)[:args.patches_per_frame]]
            out.append(p.half().cpu())
    return torch.cat(out)


@torch.no_grad()
def greedy_coreset(feats, ratio, dev, seed, proj_dim=128):
    """Greedy k-Center auf zufällig projizierten Features, gibt die Original-Features zurück."""
    n = len(feats)
    k = max(1, int(n * ratio))
    if k >= n:
        return feats.float()
    g = torch.Generator().manual_seed(seed)
    proj = (torch.randn(feats.shape[1], proj_dim, generator=g) / proj_dim ** 0.5).to(dev)
    z = torch.cat([c.to(dev).float() @ proj for c in feats.split(65536)])   # (n, proj_dim)
    sel = [int(torch.randint(n, (1,), generator=g))]
    d = (z - z[sel[0]]).norm(dim=1)
    for _ in range(k - 1):
        i = int(torch.argmax(d))
        sel.append(i)
        d = torch.minimum(d, (z - z[i]).norm(dim=1))
    return feats[sel].float()


@torch.no_grad()
def nn_dist(q, bank, chunk=4096):
    return torch.cat([torch.cdist(q[i:i + chunk].float(), bank).min(1).values
                      for i in range(0, len(q), chunk)])


def fit_banks(ds_tr, extractor, args, dev):
    bins = item_bins(ds_tr, args.phase_bins)
    banks = []
    for b in range(args.phase_bins):
        idx = np.where(bins == b)[0]
        if len(idx) == 0:
            raise RuntimeError(f"Phasenbin {b} hat keine Trainingsframes")
        feats = collect(Subset(ds_tr, idx), ds_tr, extractor, args, dev)
        banks.append(greedy_coreset(feats, args.coreset_ratio, dev, args.seed + b).cpu())
        print(f"  Bin {b}: {len(idx)} Frames, {len(feats)} Patches -> Bank {len(banks[-1])}")
        del feats
    return banks


# --------------------------------------------------------------------------- #
# Auswertung
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_frames(ds, banks, extractor, args, dev):
    P = len(banks)
    rows, maps = [], {}
    for batch in DataLoader(ds, batch_size=args.bs, num_workers=args.workers):
        x = to_imagenet(batch["x"].to(dev), ds.mean, ds.std)
        H, W = x.shape[-2:]
        f, (h, w) = extractor(x)
        bins = phase_bin(batch["phase"].numpy(), P)
        for i in range(len(f)):
            d = nn_dist(f[i].reshape(-1, f.shape[-1]), banks[bins[i]]).reshape(1, 1, h, w)
            m = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            m = gaussian_filter(m.cpu().numpy(), sigma=args.sigma)
            bid, ph = batch["bottle_id"][i], float(batch["phase"][i])
            rows.append({"bottle_id": bid, "label": int(batch["label"][i]),
                         "phase": ph, "score": float(m.max())})
            if args.save_maps:
                maps.setdefault(bid, []).append((ph, d[0, 0].half().cpu().numpy()))
    return pd.DataFrame(rows), maps


def evaluate(fr, k):
    bt = (fr.groupby("bottle_id")
            .agg(label=("label", "first"), n_frames=("score", "size"),
                 score_max=("score", "max"),
                 score_topk=("score", lambda s: s.nlargest(k).mean()))
            .reset_index())
    res = {}
    if bt.label.nunique() == 2:
        for c in ("score_max", "score_topk"):
            res[f"auroc_{c}"] = float(roc_auc_score(bt.label, bt[c]))
    return bt, res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--inputs", required=True, help="Ordner mit index.csv und .npy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-folds", type=int, nargs="+", required=True)
    ap.add_argument("--eval-split", default="trainval", choices=["trainval", "test"])
    ap.add_argument("--eval-folds", type=int, nargs="+", default=None)
    ap.add_argument("--phase-bins", type=int, default=10)
    ap.add_argument("--coreset-ratio", type=float, default=0.01)
    ap.add_argument("--patches-per-frame", type=int, default=0,
                    help="0 = alle Patches, sonst zufällige Vorauswahl (spart RAM)")
    ap.add_argument("--frame-stride-train", type=int, default=4)
    ap.add_argument("--frame-stride-eval", type=int, default=2)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-maps", action="store_true")
    ap.add_argument("--random-weights", action="store_true", help="nur zum Debuggen")
    args = ap.parse_args()

    if args.eval_split == "trainval":
        if args.eval_folds is None:
            raise SystemExit("Bitte --eval-folds angeben (sonst überlappen Training und Auswertung).")
        if set(args.eval_folds) & set(args.train_folds):
            raise SystemExit("Leakage: --eval-folds und --train-folds überlappen.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    ds_tr = BottleDataset(args.inputs, split="trainval", folds=args.train_folds, only_good=True,
                          mode="frame", frame_stride=args.frame_stride_train)
    ds_ev = BottleDataset(args.inputs, split=args.eval_split, folds=args.eval_folds,
                          mode="frame", frame_stride=args.frame_stride_eval)
    if len(ds_tr) == 0 or len(ds_ev) == 0:
        raise SystemExit(f"Leeres Dataset (train={len(ds_tr)}, eval={len(ds_ev)}) - Folds prüfen.")
    print(f"Train: {len(ds_tr)} Frames aus {len(ds_tr.df)} Gutflaschen | "
          f"Eval: {len(ds_ev)} Frames aus {len(ds_ev.df)} Flaschen | Device: {dev}")

    extractor = FeatureExtractor(random_weights=args.random_weights).to(dev).eval()

    print("Baue Memory Banks ...")
    banks = fit_banks(ds_tr, extractor, args, dev)
    torch.save({"banks": [b.half() for b in banks], "args": vars(args)}, args.out / "memory_banks.pt")

    print("Bewerte ...")
    banks = [b.to(dev) for b in banks]
    fr, maps = score_frames(ds_ev, banks, extractor, args, dev)
    bt, res = evaluate(fr, args.topk)

    fr.to_csv(args.out / "frame_scores.csv", index=False)
    bt.to_csv(args.out / "bottle_scores.csv", index=False)
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2))
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    if args.save_maps:
        (args.out / "maps").mkdir(exist_ok=True)
        for bid, lst in maps.items():
            lst.sort(key=lambda t: t[0])
            np.savez_compressed(args.out / "maps" / f"{bid}.npz",
                                phase=np.array([p for p, _ in lst]),
                                maps=np.stack([m for _, m in lst]))

    print("\nFlaschen-Scores (Maximum) nach Label:")
    print(bt.groupby("label").score_max.describe().round(3).to_string())
    print("\nMetriken:", res if res else "nur eine Klasse in der Auswertung, kein AUROC")


if __name__ == "__main__":
    main()