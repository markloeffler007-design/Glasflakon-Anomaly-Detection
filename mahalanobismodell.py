#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mahalanobis-basierte Anomalieerkennung für das Glasflakon-Projekt.

Erwartete Eingabestruktur:
    input/
      index.csv
      stats.json
      *.npy

Die .npy-Dateien enthalten Graustufen-Sequenzen mit Form (T, H, W).
Die Daten werden über die vorhandene BottleDataset-Klasse geladen.

CV:
    - ein trainval-Fold wird jeweils vollständig zurückgehalten
    - trainiert wird ausschließlich mit guten Gruppen (label == 0)
    - die guten Trainingsgruppen werden gruppenweise in Train/Validierung geteilt
    - der Validierungssatz dient zur Wahl des Anomalie-Schwellenwerts
    - der zurückgehaltene Fold wird mit guten und fehlerhaften Flakons bewertet

Test:
    - Modell auf allen guten trainval-Gruppen trainieren
    - separat auf split == "test" auswerten

Merkmale pro Pixel:
    1. Grauwert
    2. lokaler Mittelwert
    3. lokale Standardabweichung
    4. horizontaler Gradient
    5. vertikaler Gradient

Aus diesen Merkmalen wird eine regularisierte Kovarianzmatrix geschätzt.
Die quadrierte Mahalanobis-Distanz bildet die Anomaliekarte.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, sobel
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

try:
    from bottle_dataset import BottleDataset
except ImportError as exc:
    raise ImportError(
        "BottleDataset konnte nicht importiert werden. "
        "Lege bottle_dataset.py neben mahalanobismodell.py ab."
    ) from exc


DEFAULT_INPUTS = Path("input")
DEFAULT_STORE = Path("D:/Mahalanobis_Store")
DEFAULT_RESULTS = Path("D:/Mahalanobis_Results")


def group_column(df: pd.DataFrame) -> str:
    if "group_id" in df.columns:
        return "group_id"
    if "group" in df.columns:
        return "group"
    return "bottle_id"


def good_groups_for_folds(inputs_dir: Path, folds: Iterable[int]) -> list[str]:
    df = pd.read_csv(inputs_dir / "index.csv")
    col = group_column(df)
    selected = df[
        (df["split"] == "trainval")
        & (df["label"].astype(int) == 0)
        & (df["fold"].astype(int).isin(list(folds)))
    ]
    return sorted(selected[col].astype(str).unique().tolist())


def split_groups(groups: list[str], fraction: float, seed: int):
    groups = list(groups)
    if len(groups) < 2:
        raise ValueError(
            "Für die gruppenbasierte Train/Validierungs-Aufteilung werden "
            "mindestens zwei gute Gruppen benötigt."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_val = max(1, int(round(len(groups) * fraction)))
    n_val = min(n_val, len(groups) - 1)
    val_idx = set(order[:n_val].tolist())
    train = [g for i, g in enumerate(groups) if i not in val_idx]
    val = [g for i, g in enumerate(groups) if i in val_idx]
    return train, val


def make_dataset(
    inputs_dir: Path,
    split: str,
    folds: list[int] | None,
    only_good: bool,
    frame_stride: int,
    groups: set[str] | None = None,
):
    ds = BottleDataset(
        inputs_dir=inputs_dir,
        split=split,
        folds=folds,
        only_good=only_good,
        mode="frame",
        frame_stride=frame_stride,
        only_valid=True,
    )
    if groups is not None:
        col = group_column(ds.df)
        ds.items = [
            (row_i, frame_i)
            for row_i, frame_i in ds.items
            if str(ds.df.iloc[row_i][col]) in groups
        ]
    return ds


def feature_map(image: np.ndarray, window: int =   9):
    """Erzeugt (H,W,F)-Merkmale aus einem Graustufenbild in [0,1]."""
    # Das Fenster muss ungerade sein, damit der lokale Mittelwert zentriert ist.
    if window < 3 or window % 2 == 0:
        raise ValueError("feature-window muss ungerade und mindestens 3 sein.")
    image = np.asarray(image, dtype=np.float32)
    local_mean = uniform_filter(image, size=window, mode="reflect")
    local_sq_mean = uniform_filter(image * image, size=window, mode="reflect")
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    local_std = np.sqrt(local_var)
    gx = sobel(image, axis=1, mode="reflect") / 8.0
    gy = sobel(image, axis=0, mode="reflect") / 8.0
    return np.stack(
        [image, local_mean, local_std, gx, gy],
        axis=-1,
    ).astype(np.float32, copy=False)


class MahalanobisModel:
    def __init__(self, window: int =   9, reg: float = 1e-4):
        self.window = int(window)
        self.reg = float(reg)
        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    def fit(
        self,
        dataset: BottleDataset,
        stats_mean: float,
        stats_std: float,
        samples_per_frame: int,
        max_frames: int,
        seed: int,
    ):
        if len(dataset) == 0:
            raise ValueError("Der Trainingsdatensatz enthält keine Frames.")
        rng = np.random.default_rng(seed)
        frame_indices = np.arange(len(dataset))
        if len(frame_indices) > max_frames:
            frame_indices = rng.choice(frame_indices, size=max_frames, replace=False)
        samples = []
        for count, dataset_idx in enumerate(frame_indices, start=1):
            item = dataset[int(dataset_idx)]
            # BottleDataset liefert (1,H,W), normalisiert mit stats.json.
            image = item["x"].numpy()[0] * stats_std + stats_mean
            image = np.clip(image, 0.0, 1.0)
            fmap = feature_map(image, self.window)
            flat = fmap.reshape(-1, fmap.shape[-1])
            n = min(samples_per_frame, len(flat))
            chosen = rng.choice(len(flat), size=n, replace=False)
            samples.append(flat[chosen])
            if count % 100 == 0:
                print(f"  Merkmalsstichproben: {count}/{len(frame_indices)} Frames")
        x = np.concatenate(samples, axis=0).astype(np.float64)
        self.mean = x.mean(axis=0)
        covariance = np.cov(x, rowvar=False)
        # Diagonal-Regularisierung verhindert singuläre/instabile Matrizen.
        scale = float(np.trace(covariance) / covariance.shape[0])
        covariance += np.eye(covariance.shape[0]) * max(self.reg * scale, 1e-10)
        self.precision = np.linalg.pinv(covariance, hermitian=True)
        return {
            "n_training_frames_sampled": int(len(frame_indices)),
            "n_feature_vectors": int(len(x)),
            "feature_count": int(x.shape[1]),
        }

    def score(self, image: np.ndarray) -> np.ndarray:
        if self.mean is None or self.precision is None:
            raise RuntimeError("Das Mahalanobis-Modell wurde noch nicht trainiert.")
        fmap = feature_map(image, self.window).astype(np.float64)
        delta = fmap - self.mean
        # d² = (x-mu)^T * Cov^-1 * (x-mu), vektorisiert pro Pixel.
        score = np.einsum("...i,ij,...j->...", delta, self.precision, delta)
        return np.maximum(score, 0.0).astype(np.float32)

    def save(self, path: Path, metadata: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean,
            precision=self.precision,
            window=np.asarray(self.window),
            reg=np.asarray(self.reg),
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: Path):
        data = np.load(path, allow_pickle=False)
        model = cls(window=int(data["window"]), reg=float(data["reg"]))
        model.mean = data["mean"].astype(np.float64)
        model.precision = data["precision"].astype(np.float64)
        metadata = json.loads(str(data["metadata"]))
        return model, metadata


def dataset_stats(inputs_dir: Path):
    stats = json.loads((inputs_dir / "stats.json").read_text(encoding="utf-8"))
    return float(stats["mean"]), max(float(stats["std"]), 1e-8)


def score_dataset(
    model: MahalanobisModel,
    dataset: BottleDataset,
    stats_mean: float,
    stats_std: float,
    split_name: str,
):
    rows = []
    start_time = time.time()
    col = group_column(dataset.df)
    for k, (row_i, frame_i) in enumerate(dataset.items, start=1):
        item = dataset[k - 1]
        image = item["x"].numpy()[0] * stats_std + stats_mean
        image = np.clip(image, 0.0, 1.0)
        amap = model.score(image)
        # Robuster Frame-Score: Mittelwert der höchsten 1 % der Pixelwerte.
        flat = amap.reshape(-1)
        n_top = max(1, int(math.ceil(flat.size * 0.01)))
        frame_score = float(np.mean(np.partition(flat, -n_top)[-n_top:]))
        row = dataset.df.iloc[row_i]
        rows.append({
            "split": split_name,
            "bottle_id": str(row["bottle_id"]),
            "group_id": str(row[col]),
            "direction": str(row["direction"]) if "direction" in row.index else "",
            "label": int(row["label"]),
            "fold": int(row["fold"]),
            "frame_index": int(frame_i),
            "phase": float(frame_i / max(int(row["n_frames"]), 1)),
            "score": frame_score,
            "pixel_max": float(np.max(amap)),
            "pixel_mean": float(np.mean(amap)),
        })
        if k % 250 == 0 or k == len(dataset):
            elapsed = time.time() - start_time
            print(f"  {split_name}: {k}/{len(dataset)} Frames ({elapsed:.1f} s)")
    return pd.DataFrame(rows)


def aggregate_scores(frame_df: pd.DataFrame, topk: int = 5):
    if frame_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    sweep_rows = []
    for bottle_id, part in frame_df.groupby("bottle_id", sort=False):
        values = part["score"].to_numpy(dtype=float)
        k = min(topk, len(values))
        sweep_rows.append({
            "bottle_id": bottle_id,
            "group_id": str(part["group_id"].iloc[0]),
            "direction": str(part["direction"].iloc[0]),
            "label": int(part["label"].max()),
            "n_frames": int(len(part)),
            "score_max": float(np.max(values)),
            "score_topk": float(np.mean(np.partition(values, -k)[-k:])),
        })
    sweep_df = pd.DataFrame(sweep_rows)
    group_rows = []
    for group_id, part in sweep_df.groupby("group_id", sort=False):
        group_rows.append({
            "group_id": group_id,
            "label": int(part["label"].max()),
            "n_directions": int(len(part)),
            "score_dirmean": float(part["score_max"].mean()),
            "score_dirmax": float(part["score_max"].max()),
            "score_topk_dirmean": float(part["score_topk"].mean()),
            "score_topk_dirmax": float(part["score_topk"].max()),
        })
    return sweep_df, pd.DataFrame(group_rows)


def binary_metrics(y_true, scores, threshold: float):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {
        "n": int(len(y)),
        "n_good": int(np.sum(y == 0)),
        "n_bad": int(np.sum(y == 1)),
        "threshold": float(threshold),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, s))
        result["average_precision"] = float(average_precision_score(y, s))
    else:
        result["auroc"] = None
        result["average_precision"] = None
    return result


def thresholds_from_validation(val_frame_df: pd.DataFrame, quantile: float):
    """Bestimmt getrennte Schwellenwerte auf derselben Aggregationsebene."""
    thresholds = {"frame_score": float(val_frame_df["score"].quantile(quantile))}
    val_sweep, val_group = aggregate_scores(val_frame_df)
    for col in ("score_max", "score_topk"):
        if not val_sweep.empty:
            thresholds[col] = float(val_sweep[col].quantile(quantile))
    for col in ("score_dirmean", "score_dirmax", "score_topk_dirmean", "score_topk_dirmax"):
        if not val_group.empty:
            thresholds[col] = float(val_group[col].quantile(quantile))
    return thresholds


def evaluate_levels(frame_df: pd.DataFrame, thresholds: dict):
    sweep_df, group_df = aggregate_scores(frame_df)
    output = {}
    if not sweep_df.empty:
        for col in ("score_max", "score_topk"):
            output[f"sweep_{col}"] = binary_metrics(
                sweep_df["label"], sweep_df[col], thresholds[col]
            )
    if not group_df.empty:
        for col in (
            "score_dirmean", "score_dirmax",
            "score_topk_dirmean", "score_topk_dirmax",
        ):
            output[f"group_{col}"] = binary_metrics(
                group_df["label"], group_df[col], thresholds[col]
            )
    return output, sweep_df, group_df


def save_evaluation(frame_df, thresholds, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, sweep_df, group_df = evaluate_levels(frame_df, thresholds)
    frame_df.to_csv(output_dir / f"{name}_frame_scores.csv", index=False)
    sweep_df.to_csv(output_dir / f"{name}_sweep_scores.csv", index=False)
    group_df.to_csv(output_dir / f"{name}_group_scores.csv", index=False)
    (output_dir / f"{name}_metrics.json").write_text(
        json.dumps({"thresholds": thresholds, "metrics": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics


def run_cv(args):
    inputs_dir = args.inputs
    store_dir = args.store
    results_dir = args.results
    mean, std = dataset_stats(inputs_dir)
    index = pd.read_csv(inputs_dir / "index.csv")
    folds = sorted(
        int(v) for v in index.loc[index["split"] == "trainval", "fold"].unique()
        if int(v) >= 0
    )
    if len(folds) < 2:
        raise ValueError(f"Für CV werden mindestens zwei trainval-Folds benötigt: {folds}")
    all_fold_metrics = []
    all_oof_frames = []

    for eval_fold in folds:
        train_folds = [f for f in folds if f != eval_fold]
        print(f"\n{'=' * 72}\nCV Fold {eval_fold}: Training auf {train_folds}, Test auf [{eval_fold}]")
        good_groups = good_groups_for_folds(inputs_dir, train_folds)
        train_groups, val_groups = split_groups(
            good_groups, args.val_fraction, args.seed + int(eval_fold)
        )
        train_ds = make_dataset(
            inputs_dir, "trainval", train_folds, True, args.frame_stride,
            set(train_groups),
        )
        val_ds = make_dataset(
            inputs_dir, "trainval", train_folds, True, args.eval_frame_stride,
            set(val_groups),
        )
        test_ds = make_dataset(
            inputs_dir, "trainval", [eval_fold], False, args.eval_frame_stride,
        )
        print(
            f"Gute Gruppen: {len(good_groups)} | Train-Gruppen: {len(train_groups)} "
            f"| Val-Gruppen: {len(val_groups)}"
        )
        print(
            f"Frames: Train={len(train_ds)}, Validierung={len(val_ds)}, "
            f"Testfold={len(test_ds)}"
        )

        model = MahalanobisModel(args.window, args.reg)
        fit_info = model.fit(
            train_ds, mean, std, args.samples_per_frame,
            args.max_train_frames, args.seed + int(eval_fold),
        )
        val_frames = score_dataset(model, val_ds, mean, std, "validation")
        if val_frames.empty:
            raise ValueError(f"Keine Validierungsframes in Fold {eval_fold}.")
        thresholds = thresholds_from_validation(val_frames, args.threshold_quantile)
        fold_dir = results_dir / f"fold_{eval_fold}"
        model.save(
            store_dir / f"fold_{eval_fold}" / "mahalanobis_model.npz",
            {"eval_fold": eval_fold, "train_folds": train_folds, **fit_info},
        )
        test_frames = score_dataset(model, test_ds, mean, std, f"fold_{eval_fold}")
        metrics = save_evaluation(test_frames, thresholds, fold_dir, f"fold_{eval_fold}")
        all_fold_metrics.append({
            "eval_fold": eval_fold,
            "train_folds": train_folds,
            "n_train_groups": len(train_groups),
            "n_val_groups": len(val_groups),
            "n_train_frames": len(train_ds),
            "n_val_frames": len(val_ds),
            "n_test_frames": len(test_ds),
            "thresholds": thresholds,
            "fit_info": fit_info,
            "metrics": metrics,
        })
        all_oof_frames.append(test_frames)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "cv_summary.json").write_text(
        json.dumps(all_fold_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pooled = pd.concat(all_oof_frames, ignore_index=True)
    pooled.to_csv(results_dir / "oof_frame_scores.csv", index=False)
    # Für die gepoolte OOF-Rangbewertung wird kein fold-spezifischer Schwellenwert
    # vermischt; AUROC/AP werden direkt auf den Scores berechnet.
    pooled_metrics = {}
    sweep, group = aggregate_scores(pooled)
    for level_name, data, columns in (
        ("sweep", sweep, ("score_max", "score_topk")),
        ("group", group, ("score_dirmean", "score_dirmax",
                          "score_topk_dirmean", "score_topk_dirmax")),
    ):
        for col in columns:
            y = data["label"].to_numpy(dtype=int)
            s = data[col].to_numpy(dtype=float)
            if len(np.unique(y)) == 2:
                pooled_metrics[f"{level_name}_{col}"] = {
                    "auroc": float(roc_auc_score(y, s)),
                    "average_precision": float(average_precision_score(y, s)),
                    "n": int(len(y)),
                    "n_good": int(np.sum(y == 0)),
                    "n_bad": int(np.sum(y == 1)),
                }
    sweep.to_csv(results_dir / "oof_sweep_scores.csv", index=False)
    group.to_csv(results_dir / "oof_group_scores.csv", index=False)
    (results_dir / "oof_metrics.json").write_text(
        json.dumps(pooled_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nCV abgeschlossen. Ergebnisse: {results_dir}")


def run_test(args):
    inputs_dir = args.inputs
    mean, std = dataset_stats(inputs_dir)
    index = pd.read_csv(inputs_dir / "index.csv")
    folds = sorted(
        int(v) for v in index.loc[index["split"] == "trainval", "fold"].unique()
        if int(v) >= 0
    )
    good_groups = good_groups_for_folds(inputs_dir, folds)
    train_groups, val_groups = split_groups(good_groups, args.val_fraction, args.seed)
    train_ds = make_dataset(
        inputs_dir, "trainval", folds, True, args.frame_stride, set(train_groups)
    )
    val_ds = make_dataset(
        inputs_dir, "trainval", folds, True, args.eval_frame_stride, set(val_groups)
    )
    test_ds = make_dataset(
        inputs_dir, "test", None, False, args.eval_frame_stride
    )
    model = MahalanobisModel(args.window, args.reg)
    fit_info = model.fit(
        train_ds, mean, std, args.samples_per_frame,
        args.max_train_frames, args.seed,
    )
    val_frames = score_dataset(model, val_ds, mean, std, "validation")
    thresholds = thresholds_from_validation(val_frames, args.threshold_quantile)
    model.save(
        args.store / "final" / "mahalanobis_model.npz",
        {"train_folds": folds, **fit_info},
    )
    test_frames = score_dataset(model, test_ds, mean, std, "test")
    metrics = save_evaluation(test_frames, thresholds, args.results, "test")
    print(json.dumps({"thresholds": thresholds, "metrics": metrics}, indent=2, ensure_ascii=False))
    print(f"Testauswertung abgeschlossen: {args.results}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Mahalanobis-Anomalieerkennung für Glasflakon-Sequenzen"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("cv", "test"):
        p = sub.add_parser(command)
        p.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
        p.add_argument("--store", type=Path, default=DEFAULT_STORE)
        p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
        p.add_argument("--frame-stride", type=int, default=4,
                       help="Abstand der Trainingsframes (Standard: 4).")
        p.add_argument("--eval-frame-stride", type=int, default=1,
                       help="Abstand der Validierungs-/Testframes (Standard: 1).")
        p.add_argument("--window", type=int, default= 9,
                       help="Ungerade Fenstergröße für lokale Merkmale.")
        p.add_argument("--reg", type=float, default=1e-4,
                       help="Diagonal-Regularisierung der Kovarianzmatrix.")
        p.add_argument("--samples-per-frame", type=int, default=1500,
                       help="Zufällig gezogene Pixelmerkmale je Trainingsframe.")
        p.add_argument("--max-train-frames", type=int, default=1200,
                       help="Maximale Anzahl Trainingsframes für die Modellanpassung.")
        p.add_argument("--val-fraction", type=float, default=0.1)
        p.add_argument("--threshold-quantile", type=float, default=0.95)
        p.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    if args.frame_stride < 1 or args.eval_frame_stride < 1:
        raise ValueError("frame-stride und eval-frame-stride müssen mindestens 1 sein.")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction muss zwischen 0 und 1 liegen.")
    if not 0.5 < args.threshold_quantile < 1.0:
        raise ValueError("threshold-quantile muss zwischen 0.5 und 1 liegen.")
    if args.command == "cv":
        run_cv(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
