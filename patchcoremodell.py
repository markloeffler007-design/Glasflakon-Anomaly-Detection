#!/usr/bin/env python3
"""
PatchCore mit persistentem Patch-Speicher ("Vektorspeicher") statt Memory Banks, die bei jedem Lauf neu entstehen.

Zwei Schritte:

  build  Features ALLER guten Sequenzen einmal berechnen und pro Sequenz als Shard speichern
         (float16, keine Coreset-Reduktion). Bereits vorhandene Shards werden übersprungen,
         neue Sequenzen lassen sich also nachträglich ergänzen.

  score  Auswertung beliebig oft, ohne Neuaufbau. Welche Sequenzen die Bank bilden und welche Phase
         durchsucht wird, entscheidet erst die Abfrage:
           - Gruppen: nur gute Sequenzen der --train-folds (aus index.csv), nie die ausgewertete Gruppe
           - Phase:   alle Patches mit |phase - center| <= --phase-halfwidth, wobei center das
                      nächstgelegene von --phase-centers gleichmäßig verteilten Zentren ist
                      (--phase-halfwidth 1 = keine Phasenbeschränkung = Original-PatchCore)
         Die Suche ist exakt (euklidischer nächster Nachbar über alle ausgewählten Patches).

Beispiel
  python patchcoremodell.py build --inputs C:\\Glasflakons\\Input --store C:\\Glasflakons\\Store
  python patchcoremodell.py score --inputs C:\\Glasflakons\\Input --store C:\\Glasflakons\\Store \\
      --out runs\\fold4 --train-folds 0 1 2 3 --eval-folds 4 --phase-centers 20 --phase-halfwidth 0.05
  Testset (erst am Ende): --eval-split test (ohne --eval-folds)

Speicherbedarf: pro Sequenz Frames x Patches x dim x 2 Byte, bei 201 Frames, 2040 Patches und dim 1024
etwa 0,84 GB. Der Store gehört auf eine schnelle SSD (der Phasenfilter liest per Memory-Map nur
die benötigten Frames).
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


def group_col(df):
    for c in ("group_id", "group"):
        if c in df.columns:
            return c
    return "bottle_id"


# --------------------------------------------------------------------------- #
# Exakte Suche
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sq_norms(bank, chunk=65536):
    return torch.cat([(bank[i:i + chunk].float() ** 2).sum(1) for i in range(0, len(bank), chunk)])


@torch.no_grad()
def nn_dist(q, bank, b2, qchunk=1024, bchunk=32768):
    """Euklidische Distanz jedes Query-Vektors zum nächsten Bank-Vektor.
    d^2 = |q|^2 + |b|^2 - 2 q.b, in float32 gerechnet (bank liegt als float16 vor)."""
    out = []
    for i in range(0, len(q), qchunk):
        qi = q[i:i + qchunk].float()
        q2 = (qi ** 2).sum(1, keepdim=True)
        best = torch.full((len(qi),), float("inf"), device=qi.device)
        for j in range(0, len(bank), bchunk):
            g = qi @ bank[j:j + bchunk].float().T
            d2 = q2 + b2[j:j + bchunk][None, :] - 2 * g
            best = torch.minimum(best, d2.min(1).values)
        out.append(best.clamp_min(0).sqrt())
    return torch.cat(out)


# --------------------------------------------------------------------------- #
# build: Patch-Speicher aufbauen
# --------------------------------------------------------------------------- #
def cmd_build(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, store = Path(args.inputs), Path(args.store)
    (store / "shards").mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(inputs / "index.csv")
    gmap = dict(zip(index.bottle_id, index[group_col(index)]))

    mpath = store / "manifest.csv"
    manifest = (pd.read_csv(mpath) if mpath.exists() and not args.rebuild else
                pd.DataFrame(columns=["bottle_id", "group_id", "n_frames", "patches_per_frame", "dim"]))
    done = set(manifest.bottle_id)

    extractor = FeatureExtractor(args.dim, args.random_weights).to(dev).eval()
    info_path = store / "store.json"
    info = json.loads(info_path.read_text()) if info_path.exists() and not args.rebuild else None

    for split in ("trainval", "test"):  # alle guten Sequenzen, die Auswahl trifft erst score
        try:
            ds = BottleDataset(inputs, split=split, only_good=True, mode="frame",
                               frame_stride=args.frame_stride)
        except ValueError:
            continue
        todo = [k for k, (i, t) in enumerate(ds.items) if ds.df.bottle_id.iloc[i] not in done]
        if not todo:
            print(f"[{split}] nichts zu tun")
            continue
        print(f"[{split}] {len(todo)} Frames aus {len(ds.df)} guten Sequenzen")

        cur = {"id": None, "f": [], "p": []}

        def flush():
            nonlocal manifest
            if cur["id"] is None or not cur["f"]:
                return
            feats = np.stack(cur["f"])                                   # (F, P, D) float16
            np.save(store / "shards" / f"{cur['id']}.npy", feats)
            np.save(store / "shards" / f"{cur['id']}_phase.npy", np.array(cur["p"], dtype=np.float32))
            manifest = pd.concat([manifest, pd.DataFrame([{
                "bottle_id": cur["id"], "group_id": gmap.get(cur["id"], cur["id"]),
                "n_frames": feats.shape[0], "patches_per_frame": feats.shape[1], "dim": feats.shape[2]}])],
                ignore_index=True)
            manifest.to_csv(mpath, index=False)
            done.add(cur["id"])
            print(f"  gespeichert: {cur['id']} {feats.shape} ({feats.nbytes / 1e9:.2f} GB)")
            cur["f"], cur["p"] = [], []

        for batch in DataLoader(Subset(ds, todo), batch_size=args.bs, num_workers=args.workers):
            x = to_imagenet(batch["x"].to(dev), ds.mean, ds.std)
            f, (h, w) = extractor(x)
            if info is None:
                info = {"backbone": "wide_resnet50_2", "layers": "layer2+layer3", "dim": args.dim,
                        "random_weights": bool(args.random_weights),
                        "input_hw": list(x.shape[-2:]), "grid_hw": [h, w], "frame_stride": args.frame_stride}
                info_path.write_text(json.dumps(info, indent=2))
            else:
                if (info["dim"], info["random_weights"], info["input_hw"], info["frame_stride"]) != \
                        (args.dim, bool(args.random_weights), list(x.shape[-2:]), args.frame_stride):
                    raise SystemExit("Der vorhandene Store wurde mit anderen Einstellungen erstellt "
                                     "(dim, Gewichte, Bildgröße, Stride). Neuen --store wählen oder --rebuild.")
            f = f.reshape(len(f), h * w, -1).half().cpu().numpy()
            for k in range(len(f)):
                bid = batch["bottle_id"][k]
                if bid != cur["id"]:
                    flush()
                    cur["id"] = bid
                cur["f"].append(f[k])
                cur["p"].append(float(batch["phase"][k]))
        flush()
    print(f"\nStore fertig: {store}  ({len(manifest)} Sequenzen)")


# --------------------------------------------------------------------------- #
# score: aus dem Store abfragen
# --------------------------------------------------------------------------- #
def load_bank(store, shard_ids, center, halfwidth, dev):
    """Alle Patches der gewählten Sequenzen mit |phase - center| <= halfwidth."""
    parts = []
    for sid in shard_ids:
        ph = np.load(store / "shards" / f"{sid}_phase.npy")
        sel = np.where(np.abs(ph - center) <= halfwidth)[0]
        if len(sel):
            arr = np.load(store / "shards" / f"{sid}.npy", mmap_mode="r")
            parts.append(torch.from_numpy(np.ascontiguousarray(arr[sel])).reshape(-1, arr.shape[-1]))
    if not parts:
        raise RuntimeError(f"Keine Patches im Phasenfenster um {center:.3f} (+-{halfwidth})")
    bank = torch.cat(parts).to(dev)
    return bank, sq_norms(bank)


def evaluate(fr, k):
    bt = (fr.groupby("bottle_id")
            .agg(group_id=("group_id", "first"), label=("label", "first"), n_frames=("score", "size"),
                 score_max=("score", "max"),
                 score_topk=("score", lambda s: s.nlargest(k).mean()))
            .reset_index())
    gt = (bt.groupby("group_id")
            .agg(label=("label", "first"), n_sweeps=("bottle_id", "size"),
                 score_max=("score_max", "mean"), score_topk=("score_topk", "mean"))
            .reset_index())
    res = {}
    for name, d in (("sweep", bt), ("group", gt)):
        if d.label.nunique() == 2:
            for c in ("score_max", "score_topk"):
                res[f"auroc_{name}_{c}"] = float(roc_auc_score(d.label, d[c]))
    return bt, gt, res


@torch.no_grad()
def cmd_score(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, store, out = Path(args.inputs), Path(args.store), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((store / "store.json").read_text())
    manifest = pd.read_csv(store / "manifest.csv")
    index = pd.read_csv(inputs / "index.csv")
    gcol = group_col(index)

    if args.eval_split == "trainval":
        if args.eval_folds is None:
            raise SystemExit("Bitte --eval-folds angeben (sonst überlappen Bank und Auswertung).")
        if set(args.eval_folds) & set(args.train_folds):
            raise SystemExit("Leakage: --eval-folds und --train-folds überlappen.")

    ds_ev = BottleDataset(inputs, split=args.eval_split, folds=args.eval_folds,
                          mode="frame", frame_stride=args.frame_stride_eval)
    if len(ds_ev) == 0:
        raise SystemExit("Leeres Auswertungs-Dataset - Folds prüfen.")
    if list(ds_ev[0]["x"].shape[-2:]) != info["input_hw"]:
        raise SystemExit(f"Bildgröße {list(ds_ev[0]['x'].shape[-2:])} passt nicht zum Store {info['input_hw']}.")

    # Bank = gute Sequenzen der Trainings-Folds
    bank_rows = index[(index.label == 0) & (index.split == "trainval") & (index.fold.isin(args.train_folds))]
    bank_groups = set(bank_rows[gcol])
    if bank_groups & set(ds_ev.df[gcol]):
        raise SystemExit("Leakage: eine Gruppe liegt in Bank und Auswertung.")
    missing = set(bank_rows.bottle_id) - set(manifest.bottle_id)
    if missing:
        raise SystemExit(f"{len(missing)} gute Sequenzen fehlen im Store, bitte 'build' ausführen "
                         f"(z. B. {sorted(missing)[0]}).")
    shard_ids = manifest[manifest.bottle_id.isin(bank_rows.bottle_id)].bottle_id.tolist()
    if not shard_ids:
        raise SystemExit("Keine Bank-Sequenzen für diese --train-folds.")

    J = 1 if args.phase_halfwidth >= 1 else args.phase_centers
    centers = (np.arange(J) + 0.5) / J
    nf = ds_ev.df.n_frames.to_numpy()
    ph = np.array([t / nf[i] for i, t in ds_ev.items])
    assign = np.abs(ph[:, None] - centers[None, :]).argmin(1)
    order = np.argsort(assign, kind="stable")      # nach Zentrum sortieren: jede Bank wird nur einmal geladen

    print(f"Bank: {len(shard_ids)} Sequenzen aus {len(bank_groups)} Gruppen | "
          f"Eval: {len(ds_ev)} Frames aus {len(ds_ev.df)} Sequenzen | Zentren: {J} | Device: {dev}")

    extractor = FeatureExtractor(info["dim"], info["random_weights"]).to(dev).eval()
    gmap = dict(zip(index.bottle_id, index[gcol]))
    rows, maps = [], {}
    pos, cur_c, bank, b2 = 0, None, None, None
    for batch in DataLoader(Subset(ds_ev, order), batch_size=args.bs, num_workers=args.workers):
        x = to_imagenet(batch["x"].to(dev), ds_ev.mean, ds_ev.std)
        H, W = x.shape[-2:]
        f, (h, w) = extractor(x)
        for i in range(len(f)):
            c = int(assign[order[pos]])
            pos += 1
            if c != cur_c:
                bank = b2 = None
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                bank, b2 = load_bank(store, shard_ids, centers[c], args.phase_halfwidth, dev)
                cur_c = c
                print(f"  Zentrum {c + 1}/{J} (Phase {centers[c]:.3f}): Bank {len(bank):,} Patches")
            d = nn_dist(f[i].reshape(-1, f.shape[-1]), bank, b2).reshape(1, 1, h, w)
            m = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            m = gaussian_filter(m.cpu().numpy(), sigma=args.sigma)
            bid, phv = batch["bottle_id"][i], float(batch["phase"][i])
            rows.append({"bottle_id": bid, "group_id": gmap.get(bid, bid), "label": int(batch["label"][i]),
                         "phase": phv, "score": float(m.max())})
            if args.save_maps:
                maps.setdefault(bid, []).append((phv, d[0, 0].half().cpu().numpy()))

    fr = pd.DataFrame(rows)
    bt, gt, res = evaluate(fr, args.topk)
    fr.to_csv(out / "frame_scores.csv", index=False)
    bt.to_csv(out / "bottle_scores.csv", index=False)
    gt.to_csv(out / "group_scores.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    if args.save_maps:
        (out / "maps").mkdir(exist_ok=True)
        for bid, lst in maps.items():
            lst.sort(key=lambda t: t[0])
            np.savez_compressed(out / "maps" / f"{bid}.npz", phase=np.array([p for p, _ in lst]),
                                maps=np.stack([m for _, m in lst]))

    print("\nSequenz-Scores (Maximum) nach Label:")
    print(bt.groupby("label").score_max.describe().round(3).to_string())
    print("\nMetriken:", res if res else "nur eine Klasse in der Auswertung, kein AUROC")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Patch-Speicher aufbauen (einmalig bzw. inkrementell)")
    b.add_argument("--inputs", required=True, help="Ordner mit index.csv und .npy")
    b.add_argument("--store", required=True, help="Ordner für den Patch-Speicher")
    b.add_argument("--frame-stride", type=int, default=1, help="jeder n-te Frame wird gespeichert")
    b.add_argument("--dim", type=int, default=1024, help="Feature-Dimension")
    b.add_argument("--bs", type=int, default=16)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--rebuild", action="store_true", help="vorhandenen Store ignorieren und neu beginnen")
    b.add_argument("--random-weights", action="store_true", help="nur zum Debuggen")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="Auswertung gegen den Patch-Speicher")
    s.add_argument("--inputs", required=True)
    s.add_argument("--store", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--train-folds", type=int, nargs="+", required=True)
    s.add_argument("--eval-split", default="trainval", choices=["trainval", "test"])
    s.add_argument("--eval-folds", type=int, nargs="+", default=None)
    s.add_argument("--phase-centers", type=int, default=20, help="Anzahl gleichmäßig verteilter Phasenzentren")
    s.add_argument("--phase-halfwidth", type=float, default=0.05,
                   help="durchsuchte Phase = Zentrum +- Halbbreite (1 = alle Phasen)")
    s.add_argument("--frame-stride-eval", type=int, default=1)
    s.add_argument("--topk", type=int, default=5)
    s.add_argument("--sigma", type=float, default=4.0)
    s.add_argument("--bs", type=int, default=16)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--save-maps", action="store_true")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()