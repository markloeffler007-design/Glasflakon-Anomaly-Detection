#!/usr/bin/env python3
"""
PatchCore mit persistentem Patch-Speicher und Cross-Validation.

Befehle:
  build  Extrahiert Features ausschließlich aus guten trainval-Sequenzen.
  score  Wertet eine Fold-Kombination oder das unabhängige Testset aus.
  cv     Führt automatisch eine K-Fold-Cross-Validation durch und erstellt
         zusätzlich gepoolte Out-of-Fold-Metriken.

Beispiel:
  python patchcoremodell.py build --inputs ".\\Input" --store ".\\Store"

  python patchcoremodell.py cv --inputs ".\\Input" --store ".\\Store" \\
      --out ".\\Output_CV" --folds 0 1 2 3 4

  python patchcoremodell.py score --inputs ".\\Input" --store ".\\Store" \\
      --out ".\\Output_Test" --train-folds 0 1 2 3 4 --eval-split test

Wichtig:
- Der Patch-Speicher enthält nur gute Sequenzen aus trainval, niemals Testdaten.
- Bei CV wird der jeweils ausgewertete Fold aus der Trainingsbank ausgeschlossen.
- AUROC und Average Precision (AP) werden berechnet.
- Gruppen-Scores werden sowohl über den Mittelwert als auch über das Maximum
  der Scanrichtungen aggregiert. So wird sichtbar, ob eine Richtung einen
  Defekt zeigt, der durch den Mittelwert abgeschwächt würde.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Subset
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from bottle_dataset import BottleDataset

IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# Eingabe und Features
# --------------------------------------------------------------------------- #
def to_imagenet(x, mean, std):
    """Dataset-Normierung rückgängig machen und ImageNet-Normierung anwenden."""
    x = (x * std + mean).clamp(0, 1).repeat(1, 3, 1, 1)
    return (x - IMN_MEAN.to(x.device)) / IMN_STD.to(x.device)


class FeatureExtractor(torch.nn.Module):
    """WideResNet50-2: layer2 + layer3, Nachbarschaftsmittel, Dimensionsreduktion."""

    def __init__(self, out_dim=1024, random_weights=False):
        super().__init__()
        weights = None if random_weights else Wide_ResNet50_2_Weights.IMAGENET1K_V1
        model = wide_resnet50_2(weights=weights)
        self.stem = torch.nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        )
        self.l2 = model.layer2
        self.l3 = model.layer3
        self.out_dim = out_dim
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        f2 = self.l2(self.stem(x))
        f3 = self.l3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                            align_corners=False)
        features = torch.cat([f2, f3], dim=1)
        batch, channels, height, width = features.shape
        features = features.permute(0, 2, 3, 1).reshape(batch * height * width, channels)
        features = F.adaptive_avg_pool1d(
            features.unsqueeze(1), self.out_dim
        ).squeeze(1)
        return features.reshape(batch, height, width, self.out_dim), (height, width)


def group_col(df):
    for col in ("group_id", "group"):
        if col in df.columns:
            return col
    return "bottle_id"


# --------------------------------------------------------------------------- #
# Exakte Nearest-Neighbor-Suche
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sq_norms(bank, chunk=65536):
    return torch.cat([
        (bank[i:i + chunk].float() ** 2).sum(1)
        for i in range(0, len(bank), chunk)
    ])


@torch.no_grad()
def nn_dist(query, bank, bank_norms, qchunk=1024, bchunk=32768):
    """Exakte euklidische NN-Distanz, Berechnung in float32."""
    outputs = []
    for i in range(0, len(query), qchunk):
        q = query[i:i + qchunk].float()
        q2 = (q ** 2).sum(1, keepdim=True)
        best = torch.full((len(q),), float("inf"), device=q.device)
        for j in range(0, len(bank), bchunk):
            dots = q @ bank[j:j + bchunk].float().T
            d2 = q2 + bank_norms[j:j + bchunk][None, :] - 2 * dots
            best = torch.minimum(best, d2.min(1).values)
        outputs.append(best.clamp_min(0).sqrt())
    return torch.cat(outputs)


# --------------------------------------------------------------------------- #
# build: Patch-Speicher ausschließlich aus guten trainval-Sequenzen
# --------------------------------------------------------------------------- #
def cmd_build(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, store = Path(args.inputs), Path(args.store)
    (store / "shards").mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(inputs / "index.csv")
    if not {"bottle_id", "label", "split"}.issubset(index.columns):
        raise SystemExit("index.csv benötigt mindestens bottle_id, label und split.")

    # Absichtlich KEIN Testset: Testdaten dürfen nicht in den Patch-Speicher.
    train_good = index[(index["split"] == "trainval") & (index["label"] == 0)]
    if train_good.empty:
        raise SystemExit("Keine guten trainval-Sequenzen in index.csv gefunden.")

    gcol = group_col(index)
    group_map = dict(zip(index.bottle_id, index[gcol]))
    manifest_path = store / "manifest.csv"
    info_path = store / "store.json"

    if args.rebuild:
        manifest = pd.DataFrame(columns=[
            "bottle_id", "group_id", "n_frames", "patches_per_frame", "dim"
        ])
        info = None
    else:
        manifest = (pd.read_csv(manifest_path) if manifest_path.exists() else
                    pd.DataFrame(columns=[
                        "bottle_id", "group_id", "n_frames", "patches_per_frame", "dim"
                    ]))
        info = json.loads(info_path.read_text()) if info_path.exists() else None

    done = set(manifest.bottle_id.astype(str))
    extractor = FeatureExtractor(args.dim, args.random_weights).to(device).eval()

    ds = BottleDataset(inputs, split="trainval", only_good=True, mode="frame",
                       frame_stride=args.frame_stride)
    todo = [
        k for k, (row_i, _) in enumerate(ds.items)
        if str(ds.df.bottle_id.iloc[row_i]) not in done
    ]
    if not todo:
        print("[trainval] Keine neuen guten Sequenzen zu speichern.")
        print(f"Store unverändert: {store} ({len(manifest)} Sequenzen)")
        return

    print(f"[trainval] Verarbeite {len(todo)} Frames aus guten Sequenzen.")
    current = {"id": None, "features": [], "phases": []}

    def flush():
        nonlocal manifest
        if current["id"] is None or not current["features"]:
            return
        bottle_id = str(current["id"])
        feats = np.stack(current["features"]).astype(np.float16, copy=False)
        np.save(store / "shards" / f"{bottle_id}.npy", feats)
        np.save(store / "shards" / f"{bottle_id}_phase.npy",
                np.asarray(current["phases"], dtype=np.float32))
        manifest = pd.concat([manifest, pd.DataFrame([{
            "bottle_id": bottle_id,
            "group_id": group_map.get(bottle_id, bottle_id),
            "n_frames": feats.shape[0],
            "patches_per_frame": feats.shape[1],
            "dim": feats.shape[2],
        }])], ignore_index=True)
        manifest.to_csv(manifest_path, index=False)
        done.add(bottle_id)
        print(f"  gespeichert: {bottle_id}, Shape={feats.shape}, "
              f"{feats.nbytes / 1e9:.2f} GB")
        current["features"], current["phases"] = [], []

    for batch in DataLoader(Subset(ds, todo), batch_size=args.bs,
                            num_workers=args.workers):
        x = to_imagenet(batch["x"].to(device), ds.mean, ds.std)
        features, (height, width) = extractor(x)

        if info is None:
            info = {
                "backbone": "wide_resnet50_2",
                "weights": "random" if args.random_weights else "IMAGENET1K_V1",
                "layers": "layer2+layer3",
                "dim": args.dim,
                "random_weights": bool(args.random_weights),
                "input_hw": list(x.shape[-2:]),
                "grid_hw": [height, width],
                "frame_stride": args.frame_stride,
                "store_split": "trainval_only",
            }
            info_path.write_text(json.dumps(info, indent=2))
        else:
            expected = (
                info["dim"], info["random_weights"], info["input_hw"],
                info["frame_stride"]
            )
            actual = (
                args.dim, bool(args.random_weights), list(x.shape[-2:]),
                args.frame_stride
            )
            if expected != actual:
                raise SystemExit(
                    "Der vorhandene Store hat andere Einstellungen "
                    "(dim, Gewichte, Bildgröße oder Stride). "
                    "Bitte neuen --store-Ordner wählen oder mit --rebuild neu aufbauen."
                )

        features = features.reshape(len(features), height * width, -1).half().cpu().numpy()
        for k in range(len(features)):
            bottle_id = str(batch["bottle_id"][k])
            if bottle_id != current["id"]:
                flush()
                current["id"] = bottle_id
            current["features"].append(features[k])
            current["phases"].append(float(batch["phase"][k]))
    flush()
    print(f"\nStore fertig: {store} ({len(manifest)} gute trainval-Sequenzen)")
    print("Hinweis: Bereits gespeicherte Shards werden bei erneutem build übersprungen. "
          "Bei geänderten Eingabedaten bitte --rebuild verwenden.")


# --------------------------------------------------------------------------- #
# Patch-Bank laden und Metriken berechnen
# --------------------------------------------------------------------------- #
def load_bank(store, shard_ids, center, halfwidth, device):
    parts = []
    for shard_id in shard_ids:
        phases = np.load(store / "shards" / f"{shard_id}_phase.npy")
        selected = np.where(np.abs(phases - center) <= halfwidth)[0]
        if len(selected):
            arr = np.load(store / "shards" / f"{shard_id}.npy", mmap_mode="r")
            parts.append(torch.from_numpy(
                np.ascontiguousarray(arr[selected])
            ).reshape(-1, arr.shape[-1]))
    if not parts:
        raise RuntimeError(
            f"Keine Patches im Phasenfenster um {center:.3f} (+-{halfwidth})."
        )
    bank = torch.cat(parts).to(device)
    return bank, sq_norms(bank)


def add_binary_metrics(result, prefix, labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return
    result[f"auroc_{prefix}"] = float(roc_auc_score(labels, scores))
    result[f"ap_{prefix}"] = float(average_precision_score(labels, scores))


def evaluate(frame_scores, topk):
    # Kennzahlen pro bottle_id (eine Scanrichtung / Sequenz)
    sweep = (
        frame_scores.groupby("bottle_id")
        .agg(
            group_id=("group_id", "first"),
            label=("label", "first"),
            n_frames=("score", "size"),
            score_max=("score", "max"),
            score_topk=("score", lambda values: values.nlargest(topk).mean()),
        )
        .reset_index()
    )

    # Pro physischer Flasche beide Richtungs-Aggregationen ausweisen.
    # max_direction verhindert, dass ein starker Defekt-Score einer Richtung
    # durch eine unauffällige Gegenrichtung weggemittelt wird.
    group = (
        sweep.groupby("group_id")
        .agg(
            label=("label", "first"),
            n_sweeps=("bottle_id", "size"),
            score_max_mean_direction=("score_max", "mean"),
            score_max_max_direction=("score_max", "max"),
            score_topk_mean_direction=("score_topk", "mean"),
            score_topk_max_direction=("score_topk", "max"),
        )
        .reset_index()
    )

    metrics = {}
    for level, data in (("sweep", sweep), ("group", group)):
        for score_col in ("score_max", "score_topk") if level == "sweep" else (
            "score_max_mean_direction",
            "score_max_max_direction",
            "score_topk_mean_direction",
            "score_topk_max_direction",
        ):
            add_binary_metrics(
                metrics, f"{level}_{score_col}",
                data["label"], data[score_col]
            )
    return sweep, group, metrics


# --------------------------------------------------------------------------- #
# score: eine Fold-Kombination oder das Testset auswerten
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cmd_score(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, store, out = Path(args.inputs), Path(args.store), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    info_path, manifest_path = store / "store.json", store / "manifest.csv"
    if not info_path.exists() or not manifest_path.exists():
        raise SystemExit("Patch-Speicher fehlt. Bitte zuerst den Befehl 'build' ausführen.")
    info = json.loads(info_path.read_text())
    manifest = pd.read_csv(manifest_path)
    index = pd.read_csv(inputs / "index.csv")
    gcol = group_col(index)

    if args.eval_split == "trainval":
        if args.eval_folds is None:
            raise SystemExit("Für eval-split trainval muss --eval-folds angegeben werden.")
        if set(args.eval_folds) & set(args.train_folds):
            raise SystemExit("Leakage: --eval-folds und --train-folds überlappen.")
    elif args.eval_folds is not None:
        raise SystemExit("--eval-folds ist nur bei --eval-split trainval erlaubt.")

    ds_eval = BottleDataset(
        inputs, split=args.eval_split, folds=args.eval_folds,
        mode="frame", frame_stride=args.frame_stride_eval
    )
    if len(ds_eval) == 0:
        raise SystemExit("Leeres Auswertungs-Dataset - Split/Folds prüfen.")
    if list(ds_eval[0]["x"].shape[-2:]) != info["input_hw"]:
        raise SystemExit(
            f"Bildgröße {list(ds_eval[0]['x'].shape[-2:])} "
            f"passt nicht zum Store {info['input_hw']}."
        )

    bank_rows = index[
        (index.label == 0) &
        (index.split == "trainval") &
        (index.fold.isin(args.train_folds))
    ]
    bank_groups = set(bank_rows[gcol])
    eval_groups = set(ds_eval.df[gcol])
    if bank_groups & eval_groups:
        raise SystemExit("Leakage: Eine Gruppe liegt gleichzeitig in Bank und Auswertung.")

    missing = set(bank_rows.bottle_id.astype(str)) - set(manifest.bottle_id.astype(str))
    if missing:
        example = sorted(missing)[0]
        raise SystemExit(
            f"{len(missing)} gute Trainingssequenzen fehlen im Store "
            f"(z. B. {example}). Bitte 'build' ausführen."
        )
    shard_ids = manifest[
        manifest.bottle_id.astype(str).isin(bank_rows.bottle_id.astype(str))
    ].bottle_id.astype(str).tolist()
    if not shard_ids:
        raise SystemExit("Keine Bank-Sequenzen für diese --train-folds.")

    num_centers = 1 if args.phase_halfwidth >= 1 else args.phase_centers
    centers = (np.arange(num_centers) + 0.5) / num_centers
    n_frames = ds_eval.df.n_frames.to_numpy()
    phases = np.array([t / n_frames[i] for i, t in ds_eval.items])
    assignment = np.abs(phases[:, None] - centers[None, :]).argmin(1)
    order = np.argsort(assignment, kind="stable")

    print(
        f"Bank: {len(shard_ids)} Sequenzen aus {len(bank_groups)} Gruppen | "
        f"Eval: {len(ds_eval)} Frames aus {len(ds_eval.df)} Sequenzen | "
        f"Phasenzentren: {num_centers} | Device: {device}"
    )
    extractor = FeatureExtractor(info["dim"], info["random_weights"]).to(device).eval()
    group_map = dict(zip(index.bottle_id.astype(str), index[gcol]))
    rows, saved_maps = [], {}
    position, current_center = 0, None
    bank = bank_norms = None

    for batch in DataLoader(
        Subset(ds_eval, order), batch_size=args.bs, num_workers=args.workers
    ):
        x = to_imagenet(batch["x"].to(device), ds_eval.mean, ds_eval.std)
        image_h, image_w = x.shape[-2:]
        features, (height, width) = extractor(x)

        for i in range(len(features)):
            center_idx = int(assignment[order[position]])
            position += 1
            if center_idx != current_center:
                bank = bank_norms = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                bank, bank_norms = load_bank(
                    store, shard_ids, centers[center_idx],
                    args.phase_halfwidth, device
                )
                current_center = center_idx
                print(
                    f"  Zentrum {center_idx + 1}/{num_centers} "
                    f"(Phase {centers[center_idx]:.3f}): {len(bank):,} Patches"
                )

            distance = nn_dist(
                features[i].reshape(-1, features.shape[-1]), bank, bank_norms
            ).reshape(1, 1, height, width)
            anomaly_map = F.interpolate(
                distance, size=(image_h, image_w),
                mode="bilinear", align_corners=False
            )[0, 0]
            anomaly_map = gaussian_filter(
                anomaly_map.cpu().numpy(), sigma=args.sigma
            )

            bottle_id = str(batch["bottle_id"][i])
            phase_value = float(batch["phase"][i])
            rows.append({
                "bottle_id": bottle_id,
                "group_id": group_map.get(bottle_id, bottle_id),
                "label": int(batch["label"][i]),
                "phase": phase_value,
                "score": float(anomaly_map.max()),
            })
            if args.save_maps:
                saved_maps.setdefault(bottle_id, []).append(
                    (phase_value, distance[0, 0].half().cpu().numpy())
                )

    frame_df = pd.DataFrame(rows)
    sweep_df, group_df, metrics = evaluate(frame_df, args.topk)

    frame_df.to_csv(out / "frame_scores.csv", index=False)
    sweep_df.to_csv(out / "bottle_scores.csv", index=False)
    group_df.to_csv(out / "group_scores.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.json").write_text(json.dumps(
        {key: str(value) for key, value in vars(args).items()}, indent=2
    ))

    if args.save_maps:
        maps_dir = out / "maps"
        maps_dir.mkdir(exist_ok=True)
        for bottle_id, items in saved_maps.items():
            items.sort(key=lambda item: item[0])
            np.savez_compressed(
                maps_dir / f"{bottle_id}.npz",
                phase=np.array([phase for phase, _ in items]),
                maps=np.stack([map_array for _, map_array in items])
            )

    print("\nSequenz-Scores nach Label:")
    print(sweep_df.groupby("label")[["score_max", "score_topk"]]
          .describe().round(3).to_string())
    print("\nMetriken (AUROC und AP):")
    print(json.dumps(metrics, indent=2) if metrics else
          "Nur eine Klasse in der Auswertung; AUROC/AP nicht definiert.")


# --------------------------------------------------------------------------- #
# cv: alle Folds automatisch durchlaufen und OOF-Ergebnisse zusammenführen
# --------------------------------------------------------------------------- #
def cmd_cv(args):
    folds = list(args.folds)
    if len(folds) < 2 or len(set(folds)) != len(folds):
        raise SystemExit("--folds muss mindestens zwei unterschiedliche Fold-IDs enthalten.")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    all_groups = []

    for eval_fold in folds:
        train_folds = [fold for fold in folds if fold != eval_fold]
        fold_out = out_root / f"fold_{eval_fold}"
        print("\n" + "=" * 72)
        print(f"CV Fold {eval_fold}: Training={train_folds}, Evaluation={[eval_fold]}")
        print("=" * 72)

        score_args = argparse.Namespace(
            inputs=args.inputs,
            store=args.store,
            out=str(fold_out),
            train_folds=train_folds,
            eval_split="trainval",
            eval_folds=[eval_fold],
            phase_centers=args.phase_centers,
            phase_halfwidth=args.phase_halfwidth,
            frame_stride_eval=args.frame_stride_eval,
            topk=args.topk,
            sigma=args.sigma,
            bs=args.bs,
            workers=args.workers,
            save_maps=args.save_maps,
        )
        cmd_score(score_args)

        group_file = fold_out / "group_scores.csv"
        if not group_file.exists():
            raise SystemExit(f"Erwartete Fold-Ergebnisdatei fehlt: {group_file}")
        fold_groups = pd.read_csv(group_file)
        fold_groups["eval_fold"] = eval_fold
        all_groups.append(fold_groups)

    pooled = pd.concat(all_groups, ignore_index=True)
    pooled.to_csv(out_root / "oof_group_scores.csv", index=False)

    pooled_metrics = {}
    for score_col in (
        "score_max_mean_direction",
        "score_max_max_direction",
        "score_topk_mean_direction",
        "score_topk_max_direction",
    ):
        add_binary_metrics(
            pooled_metrics, f"group_{score_col}",
            pooled["label"], pooled[score_col]
        )

    (out_root / "oof_metrics.json").write_text(
        json.dumps(pooled_metrics, indent=2)
    )
    print("\n" + "=" * 72)
    print("Gepoolte Out-of-Fold-Metriken über alle ausgewerteten Gruppen")
    print(json.dumps(pooled_metrics, indent=2))
    print(f"\nErgebnisse gespeichert unter: {out_root}")
    print(
        "Hinweis: Gepoolte OOF-Metriken sind zusätzlich zu den Fold-Metriken "
        "zu betrachten. Da jeder Fold eine andere Trainingsbank besitzt, "
        "können die absoluten Score-Skalen zwischen Folds variieren."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_score_arguments(parser):
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-folds", type=int, nargs="+", required=True)
    parser.add_argument("--eval-split", default="trainval",
                        choices=["trainval", "test"])
    parser.add_argument("--eval-folds", type=int, nargs="+", default=None)
    parser.add_argument("--phase-centers", type=int, default=20)
    parser.add_argument("--phase-halfwidth", type=float, default=0.05)
    parser.add_argument("--frame-stride-eval", type=int, default=1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-maps", action="store_true")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    build = subparsers.add_parser("build", help="Patch-Speicher aus guten trainval-Daten bauen")
    build.add_argument("--inputs", required=True)
    build.add_argument("--store", required=True)
    build.add_argument("--frame-stride", type=int, default=1)
    build.add_argument("--dim", type=int, default=1024)
    build.add_argument("--bs", type=int, default=16)
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--rebuild", action="store_true")
    build.add_argument("--random-weights", action="store_true")
    build.set_defaults(func=cmd_build)

    score = subparsers.add_parser("score", help="Eine Fold-Kombination oder Testdaten auswerten")
    add_score_arguments(score)
    score.set_defaults(func=cmd_score)

    cv = subparsers.add_parser("cv", help="Automatische K-Fold-Cross-Validation")
    cv.add_argument("--inputs", required=True)
    cv.add_argument("--store", required=True)
    cv.add_argument("--out", required=True)
    cv.add_argument("--folds", type=int, nargs="+", required=True)
    cv.add_argument("--phase-centers", type=int, default=20)
    cv.add_argument("--phase-halfwidth", type=float, default=0.05)
    cv.add_argument("--frame-stride-eval", type=int, default=1)
    cv.add_argument("--topk", type=int, default=5)
    cv.add_argument("--sigma", type=float, default=4.0)
    cv.add_argument("--bs", type=int, default=16)
    cv.add_argument("--workers", type=int, default=4)
    cv.add_argument("--save-maps", action="store_true")
    cv.set_defaults(func=cmd_cv)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
