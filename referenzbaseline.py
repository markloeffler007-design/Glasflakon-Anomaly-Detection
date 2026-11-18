#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Referenz-Baseline "Phasen-Template" für die Glasflakon-Inspektion.

Idee: Alle guten Fahrten laufen entlang derselben Strecke. Für jede Fahrtrichtung
und jeden Phasen-Bin wird aus den guten Trainingsframes ein Template berechnet
(Mittelwert und Standardabweichung je Pixel). Ein neuer Frame wird mit dem Template
seiner Richtung und Phase verglichen:

    Karte = | Frame - Mittelwert | / max(Std, std_floor)        (z-Wert je Pixel)

Optional mit Verschiebungstoleranz (--shift-tol): je Pixel wird das Minimum über alle
Verschiebungen des Templates um bis zu +-t Pixel genommen, damit kleine
Positionsabweichungen keine Fehlalarme an Kanten erzeugen. Danach Gauß-Glättung,
Frame-Score = Maximum (oder Mittel der höchsten X % Pixel).

Auswertung identisch zu efficientadmodell.py: Frame -> Sweep -> Gruppe, score und
score_z (phasenkalibriert), AUROC/AP, Schwellen nur aus held-out guten Gruppen.
Es wird nichts trainiert; "build" schätzt nur die Templates.

Befehle
    build   Templates aus allen guten trainval-Folds (Modellname "final").
    score   Bewertet ein Modell auf dem Testset oder auf nicht benutzten Folds.
    cv      K-Fold-Cross-Validation (je Fold eigene Templates).

Beispiele
    python referenzbaseline.py build --inputs "C:/Glasflakons/Input" --store "D:/Baseline_Store"
    python referenzbaseline.py score --inputs "C:/Glasflakons/Input" --store "D:/Baseline_Store" --split test
    python referenzbaseline.py cv    --inputs "C:/Glasflakons/Input" --store "D:/Baseline_Store"
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset

from bottle_dataset import BottleDataset


# ============================================================
# KONFIGURATION
# ============================================================

MIN_SCALE_FRACTION = 0.25
SCORE_COLS = ("score", "score_z", "score_knn")
SWEEP_COMBOS = [f"{c}_{a}" for c in SCORE_COLS for a in ("max", "topk")]
GROUP_COMBOS = [f"{s}_{d}" for s in SWEEP_COMBOS for d in ("dirmean", "dirmax")]

MIN_BIN_COUNT = 3   # Bins mit weniger Frames übernehmen die Statistik des nächsten Bins


# ============================================================
# GEMEINSAME AUSWERTUNG (identisch zu efficientadmodell.py)
# ============================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_col(df):
    for col in ("group_id", "group"):
        if col in df.columns:
            return col
    return "bottle_id"


class FrameDataset(Dataset):
    """
    Lazy Zugriff auf Einzelframes über BottleDataset (mode="frame").

    Es wird nichts vorab in den Speicher geladen. BottleDataset liefert
    normalisierte Graustufenbilder; die Normalisierung wird hier rückgängig
    gemacht, da EfficientAD Werte in [0, 1] erwartet (ImageNet-Normierung
    erfolgt im Modell).

    groups / exclude_groups filtern nach group_id (bzw. bottle_id, falls es
    keine Gruppenspalte gibt).
    """

    def __init__(
        self, inputs_dir, split, folds=None, only_good=False,
        frame_stride=1, groups=None, exclude_groups=None,
        add_defects_from_test=False,
    ):
        # (split, folds, only_good, nur_fehlerhafte)
        specs = [(split, folds, only_good, False)]
        if add_defects_from_test and split != "test":
            # zusätzlich alle fehlerhaften Gruppen des Testsets (nie trainiert)
            specs.append(("test", None, False, True))

        self.parts = []   # BottleDataset je Teilmenge
        self.items = []   # (part, lokaler Index, Frame t)
        for sp, fo, og, only_bad in specs:
            try:
                base = BottleDataset(
                    inputs_dir, split=sp, folds=fo, only_good=og,
                    mode="frame", frame_stride=frame_stride, only_valid=True,
                )
            except ValueError:
                if only_bad:
                    continue          # Testset ohne Daten: nichts zu ergänzen
                raise

            col = group_col(base.df)
            group_of_row = base.df[col].astype(str).to_numpy()
            label_of_row = base.df["label"].to_numpy()

            keep = []
            for row_i, t in base.items:
                g = group_of_row[row_i]
                if only_bad and label_of_row[row_i] != 1:
                    continue
                if groups is not None and g not in groups:
                    continue
                if exclude_groups is not None and g in exclude_groups:
                    continue
                keep.append((row_i, t))

            base.items = keep
            part = len(self.parts)
            self.parts.append(base)
            self.items.extend((part, i, t) for i, (_, t) in enumerate(keep))

        if not self.items:
            raise RuntimeError(
                f"Keine Frames für split={split}, folds={folds}, only_good={only_good} "
                f"nach Gruppenfilter."
            )
        self.mean = self.parts[0].mean
        self.std = self.parts[0].std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        part, local_i, t = self.items[k]
        s = self.parts[part][local_i]
        gray = (s["x"][0] * self.std + self.mean).clamp(0.0, 1.0)  # (H, W)
        return {
            "gray": gray,
            "label": int(s["label"]),
            "bottle_id": str(s["bottle_id"]),
            "group_id": str(s.get("group_id", s["bottle_id"])),
            "direction": str(s.get("direction", "all")),
            "phase": float(s["phase"]),
            "frame_index": int(t),
        }

    def frame_hw(self):
        return tuple(self[0]["gray"].shape)


def list_good_groups(inputs_dir, folds):
    index = pd.read_csv(Path(inputs_dir) / "index.csv")
    col = group_col(index)
    sel = (
        (index["split"] == "trainval")
        & (index["label"] == 0)
        & (index["fold"].isin(folds))
    )
    return sorted(index.loc[sel, col].astype(str).unique().tolist())


def split_train_val_groups(good_groups, val_fraction, seed):
    """Teilt gute Gruppen in Trainings- und Validierungsgruppen (nach group_id)."""
    if len(good_groups) < 2:
        raise SystemExit(
            f"Nur {len(good_groups)} gute Gruppe(n) in den Trainings-Folds; "
            "mindestens 2 nötig (Training + Validierung)."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(good_groups))
    n_val = int(round(val_fraction * len(good_groups)))
    n_val = min(max(1, n_val), len(good_groups) - 1)
    val = sorted(good_groups[i] for i in order[:n_val])
    train = sorted(good_groups[i] for i in order[n_val:])
    return train, val


def robust_stats(values):
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    scale = float(np.median(np.abs(values - med)) * 1.4826)
    return med, max(scale, 1e-6)


class PhaseCalibration:
    """
    Robuste Standardisierung der Frame-Scores je (Richtung, Phasen-Bin) mit den
    Statistiken guter Validierungsframes.

    Hintergrund: Das Durchlichtbild einer guten Flasche ist je nach Position sehr
    unterschiedlich komplex, die Roh-Scores haben daher phasenabhängige Niveaus.
    Zellen mit zu wenigen Frames fallen auf die globalen Statistiken zurück.
    bins <= 0 schaltet die Kalibrierung ab (score_z == score).
    """

    def __init__(self, bins, min_count=8):
        self.bins = int(bins)
        self.min_count = int(min_count)
        self.global_stats = [0.0, 1.0]
        self.cells = {}

    def _key(self, direction, phase):
        b = min(int(max(phase, 0.0) * self.bins), self.bins - 1)
        return f"{direction}|{b}"

    def fit(self, df):
        if self.bins <= 0:
            return self
        med, scale = robust_stats(df["score"])
        self.global_stats = [med, scale]
        keys = [self._key(d, p) for d, p in zip(df["direction"], df["phase"])]
        for key, sub in df.assign(_key=keys).groupby("_key"):
            if len(sub) >= self.min_count:
                c_med, c_scale = robust_stats(sub["score"])
                self.cells[key] = [c_med, max(c_scale, MIN_SCALE_FRACTION * scale)]
        return self

    def transform(self, df):
        if self.bins <= 0:
            return df["score"].to_numpy(dtype=float).copy()
        out = []
        for d, p, s in zip(df["direction"], df["phase"], df["score"]):
            med, scale = self.cells.get(self._key(d, p), self.global_stats)
            out.append((s - med) / scale)
        return np.asarray(out, dtype=float)

    def to_dict(self):
        return {
            "bins": self.bins,
            "min_count": self.min_count,
            "global_stats": self.global_stats,
            "cells": self.cells,
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(data["bins"], data["min_count"])
        obj.global_stats = data["global_stats"]
        obj.cells = data["cells"]
        return obj


def knn_scores(frames, reference, bins, k):
    """Mittlerer absoluter Rohscore-Abstand innerhalb derselben Richtung und Phase.

    Alle Frames der eigenen group_id bleiben ausgeschlossen, auch bei --allow-leakage.
    Bei bins=0 wird nur nach Richtung getrennt.
    """
    if k < 1:
        raise ValueError("k muss positiv sein.")
    if not reference["label"].eq(0).all():
        raise ValueError("Die kNN-Referenz darf nur gute Validierungsframes enthalten.")

    def with_bins(df):
        phase = df["phase"].clip(0, 1)
        count = max(1, bins)
        return df.assign(phase_bin=np.minimum((phase * count).astype(int), count - 1))

    keys = ["direction", "phase_bin"]
    pools = dict(tuple(with_bins(reference).groupby(keys, sort=False)))
    queries = with_bins(frames.reset_index(drop=True))
    result = np.empty(len(queries), dtype=float)
    for key, cell in queries.groupby(keys, sort=False):
        pool = pools.get(key)
        if pool is None:
            raise ValueError(f"Keine guten kNN-Referenzen für {key}. Phasen-Bins oder Stride reduzieren.")
        # Neue Testflakons können gemeinsam dieselbe Referenz durchsuchen.
        overlap = cell["group_id"].isin(pool["group_id"])
        exclusions = cell["group_id"].where(overlap, "")
        for excluded, batch in cell.groupby(exclusions, sort=False):
            available = pool.loc[pool["group_id"] != excluded, "score"]
            if available.empty:
                raise ValueError(f"Für {key} fehlt eine zweite Validierungsgruppe. "
                                 "--val-fraction erhöhen oder --phase-bins reduzieren.")
            n = min(k, len(available))
            tree = cKDTree(available.to_numpy(dtype=float).reshape(-1, 1))
            for start in range(0, len(batch), max(1, 100000 // n)):
                part = batch.iloc[start:start + max(1, 100000 // n)]
                distances, _ = tree.query(part[["score"]].to_numpy(), k=n)
                result[part.index] = distances.reshape(len(part), n).mean(axis=1)
    return result


def save_heatmap(gray, amap, vmax, path):
    """Speichert Frame und Anomaliekarte nebeneinander (feste Skala 0..vmax)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_u8 = (gray.numpy() * 255.0).astype(np.uint8)
    heat_u8 = (np.clip(amap / max(vmax, 1e-8), 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(np.hstack([frame_u8, heat_u8])).save(path)


def aggregate(frame_df, topk):
    """Frames -> Sweeps (bottle_id) -> Gruppen (group_id)."""
    sweeps = []
    for bottle_id, g in frame_df.groupby("bottle_id", sort=True):
        row = {
            "bottle_id": bottle_id,
            "group_id": g["group_id"].iloc[0],
            "direction": g["direction"].iloc[0],
            "label": int(g["label"].iloc[0]),
            "n_frames": int(len(g)),
        }
        for col in SCORE_COLS:
            if col not in g:
                continue
            v = np.sort(g[col].to_numpy(dtype=float))[::-1]
            row[f"{col}_max"] = float(v[0])
            row[f"{col}_topk"] = float(v[:topk].mean())
        sweeps.append(row)
    sweep_df = pd.DataFrame(sweeps)

    groups = []
    for group_id, g in sweep_df.groupby("group_id", sort=True):
        row = {
            "group_id": group_id,
            "label": int(g["label"].iloc[0]),
            "n_sweeps": int(len(g)),
        }
        for combo in SWEEP_COMBOS:
            if combo not in g:
                continue
            row[f"{combo}_dirmean"] = float(g[combo].mean())
            row[f"{combo}_dirmax"] = float(g[combo].max())
        groups.append(row)
    return sweep_df, pd.DataFrame(groups)


def binary_metrics(labels, scores, threshold=None):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    out = {"auroc": None, "ap": None}

    if len(np.unique(labels)) == 2:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["ap"] = float(average_precision_score(labels, scores))

    if threshold is not None:
        pred = scores > threshold
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        fpr = fp / (fp + tn) if fp + tn else None
        f1 = (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn) and (tp + fn) else None
        )
        out.update({
            "threshold": float(threshold),
            "precision": precision, "recall": recall, "fpr": fpr, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return out


def compute_metrics(sweep_df, group_df, thresholds=None):
    out = {
        "n_sweeps": int(len(sweep_df)),
        "n_sweeps_good": int((sweep_df["label"] == 0).sum()),
        "n_sweeps_bad": int((sweep_df["label"] == 1).sum()),
        "n_groups": int(len(group_df)),
        "n_groups_good": int((group_df["label"] == 0).sum()),
        "n_groups_bad": int((group_df["label"] == 1).sum()),
        "sweep": {},
        "group": {},
    }
    for level, df, combos in (
        ("sweep", sweep_df, SWEEP_COMBOS),
        ("group", group_df, GROUP_COMBOS),
    ):
        for combo in combos:
            if combo not in df:
                continue
            thr = None if thresholds is None else thresholds.get(level, {}).get(combo)
            out[level][combo] = binary_metrics(df["label"], df[combo], thr)
    return out


def metrics_table(metrics):
    rows = []
    for level in ("sweep", "group"):
        for combo, m in metrics[level].items():
            rows.append({"level": level, "score": combo, **m})
    return pd.DataFrame(rows)


def print_metrics(metrics):
    table = metrics_table(metrics)
    cols = [c for c in ("level", "score", "auroc", "ap", "precision", "recall", "fpr", "f1")
            if c in table.columns]
    print(
        f"\nSweeps: {metrics['n_sweeps']} "
        f"(gut {metrics['n_sweeps_good']} / Fehler {metrics['n_sweeps_bad']}) | "
        f"Gruppen: {metrics['n_groups']} "
        f"(gut {metrics['n_groups_good']} / Fehler {metrics['n_groups_bad']})"
    )
    print(table[cols].to_string(
        index=False,
        float_format=lambda v: f"{v:.3f}",
        na_rep="-",
    ))


def thresholds_from_validation(val_frame_df, topk, quantile):
    """Schwellen je Score-Kombination aus GUTEN Validierungsdaten."""
    sweep_df, group_df = aggregate(val_frame_df, topk)
    thresholds = {"sweep": {}, "group": {}}
    for level, df, combos in (
        ("sweep", sweep_df, SWEEP_COMBOS),
        ("group", group_df, GROUP_COMBOS),
    ):
        for combo in combos:
            if combo not in df:
                continue
            thresholds[level][combo] = float(np.quantile(df[combo].to_numpy(), quantile))
    return thresholds, len(sweep_df), len(group_df)


def trainval_folds(inputs_dir):
    index = pd.read_csv(Path(inputs_dir) / "index.csv")
    folds = index.loc[index["split"] == "trainval", "fold"].dropna().unique().tolist()
    return sorted(int(f) for f in folds)


# ============================================================
# TEMPLATE-MODELL
# ============================================================

class PhaseTemplateModel:
    """Pixelweise Templates (Mittelwert, Std) je Richtung und Phasen-Bin."""

    def __init__(self, bins, std_floor, shift_tol):
        self.bins = int(bins)
        self.std_floor = float(std_floor)
        self.shift_tol = int(shift_tol)
        self.templates = {}   # direction -> (mean[B,H,W], std[B,H,W]) float32
        self.fill_info = {}

    def bin_of(self, phase):
        return min(int(max(phase, 0.0) * self.bins), self.bins - 1)

    def fit(self, frames):
        h, w = frames.frame_hw()
        sums, sqs, counts = {}, {}, {}
        total = len(frames)
        for k in range(total):
            s = frames[k]
            d = s["direction"]
            if d not in sums:
                sums[d] = np.zeros((self.bins, h, w), dtype=np.float64)
                sqs[d] = np.zeros((self.bins, h, w), dtype=np.float64)
                counts[d] = np.zeros(self.bins, dtype=np.int64)
            b = self.bin_of(s["phase"])
            x = s["gray"].numpy().astype(np.float64)
            sums[d][b] += x
            sqs[d][b] += x * x
            counts[d][b] += 1
            if (k + 1) % 500 == 0 or k + 1 == total:
                print(f"  Template-Statistik [{k + 1}/{total}]")

        for d in sums:
            n = counts[d]
            valid = np.where(n >= MIN_BIN_COUNT)[0]
            if len(valid) == 0:
                raise SystemExit(
                    f"Richtung '{d}': kein Phasen-Bin mit mindestens {MIN_BIN_COUNT} "
                    f"Frames. --bins verkleinern."
                )
            mean = np.zeros((self.bins, h, w), dtype=np.float32)
            std = np.zeros((self.bins, h, w), dtype=np.float32)
            filled = 0
            for b in range(self.bins):
                src = b
                if n[b] < MIN_BIN_COUNT:
                    src = int(valid[np.argmin(np.abs(valid - b))])
                    filled += 1
                m = sums[d][src] / n[src]
                var = np.maximum(sqs[d][src] / n[src] - m * m, 0.0)
                mean[b] = m.astype(np.float32)
                std[b] = np.sqrt(var).astype(np.float32)
            self.templates[d] = (mean, std)
            self.fill_info[d] = {
                "frames": int(n.sum()),
                "min_frames_per_bin": int(n.min()),
                "median_frames_per_bin": float(np.median(n)),
                "bins_aufgefuellt": int(filled),
            }
            print(f"  Richtung {d}: {int(n.sum())} Frames, Frames je Bin "
                  f"min {int(n.min())} / Median {float(np.median(n)):.0f}, "
                  f"aufgefüllte Bins: {filled}")
        return self

    def anomaly_map(self, gray, direction, phase):
        """z-Wert-Karte (H, W); bei shift_tol > 0 Minimum über Verschiebungen."""
        if direction not in self.templates:
            raise SystemExit(
                f"Richtung '{direction}' wurde im Training nicht gesehen "
                f"(bekannt: {sorted(self.templates)})."
            )
        mean, std = self.templates[direction]
        b = self.bin_of(phase)
        mu = mean[b]
        sd = np.maximum(std[b], self.std_floor)
        x = gray.numpy().astype(np.float32)
        t = self.shift_tol
        if t <= 0:
            return np.abs(x - mu) / sd
        h, w = x.shape
        pad_mu = np.pad(mu, t, mode="edge")
        pad_sd = np.pad(sd, t, mode="edge")
        best = None
        for dy in range(-t, t + 1):
            for dx in range(-t, t + 1):
                m = pad_mu[t + dy:t + dy + h, t + dx:t + dx + w]
                s = pad_sd[t + dy:t + dy + h, t + dx:t + dx + w]
                z = np.abs(x - m) / s
                best = z if best is None else np.minimum(best, z)
        return best

    def save(self, path):
        arrays = {}
        for d, (mean, std) in self.templates.items():
            arrays[f"{d}__mean"] = mean
            arrays[f"{d}__std"] = std
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path, bins, std_floor, shift_tol):
        obj = cls(bins, std_floor, shift_tol)
        with np.load(path) as data:
            names = sorted({k.rsplit("__", 1)[0] for k in data.files})
            for d in names:
                obj.templates[d] = (data[f"{d}__mean"], data[f"{d}__std"])
        return obj


def frame_score(amap, pixel_percent):
    if pixel_percent <= 0:
        return float(amap.max())
    flat = amap.reshape(-1)
    n = max(1, int(round(flat.size * pixel_percent / 100.0)))
    return float(np.partition(flat, flat.size - n)[-n:].mean())


def score_frames(model, frames, cfg, heatmap_dir=None, heatmap_vmax=None, label=""):
    rows = []
    total = len(frames)
    start = time.perf_counter()
    for k in range(total):
        t0 = time.perf_counter()
        sample = frames[k]
        amap = model.anomaly_map(sample["gray"], sample["direction"], sample["phase"])
        if cfg["sigma"] > 0:
            amap = gaussian_filter(amap, sigma=cfg["sigma"])
        score = frame_score(amap, cfg["pixel_percent"])
        rows.append({
            "frame_index": sample["frame_index"],
            "bottle_id": sample["bottle_id"],
            "group_id": sample["group_id"],
            "direction": sample["direction"],
            "phase": sample["phase"],
            "label": sample["label"],
            "score": score,
            "inference_time": time.perf_counter() - t0,
        })
        if heatmap_dir is not None and heatmap_vmax is not None:
            safe = sample["bottle_id"].replace("/", "_").replace("\\", "_")
            save_heatmap(
                sample["gray"], amap, heatmap_vmax,
                Path(heatmap_dir) / f"{safe}_{sample['frame_index']:06d}.png",
            )
        if k == 0 or (k + 1) % 200 == 0 or k + 1 == total:
            print(f"  {label}[{k + 1}/{total}] Score={score:.3f} "
                  f"({time.perf_counter() - start:.0f} s gesamt)")
    return pd.DataFrame(rows)


# ============================================================
# BUILD / SCORE / CV
# ============================================================

def make_cfg(args, frame_hw, train_folds, train_groups, val_groups, train_time, fill_info):
    return {
        "model": "PhaseTemplateBaseline",
        "knn_k": args.knn_k,
        "bins": args.bins,
        "std_floor": args.std_floor,
        "shift_tol": args.shift_tol,
        "sigma": args.sigma,
        "pixel_percent": args.pixel_percent,
        "topk": args.topk,
        "frame_stride": args.frame_stride,
        "eval_frame_stride": args.eval_frame_stride or 1,
        "threshold_quantile": args.threshold_quantile,
        "frame_hw": list(frame_hw),
        "train_folds": [int(f) for f in train_folds],
        "train_groups": train_groups,
        "val_groups": val_groups,
        "seed": args.seed,
        "train_time_seconds": train_time,
        "template_info": fill_info,
    }


def train_model(inputs_dir, store_dir, train_folds, model_name, args):
    store_dir = ensure_dir(store_dir)
    model_dir = ensure_dir(store_dir / model_name)

    print("\n" + "=" * 70)
    print(f"PHASEN-TEMPLATE  ({model_name})")
    print("=" * 70)

    good_groups = list_good_groups(inputs_dir, train_folds)
    train_groups, val_groups = split_train_val_groups(
        good_groups, args.val_fraction, args.seed
    )
    print(f"Train-Folds: {train_folds}")
    print(f"Gute Gruppen: {len(good_groups)} -> Templates {len(train_groups)}, "
          f"Validierung {len(val_groups)}")
    if args.knn_k < 0:
        raise SystemExit("--knn-k muss mindestens 0 sein.")
    if args.knn_k and len(val_groups) < 2:
        raise SystemExit("kNN benötigt mindestens zwei gute Validierungsgruppen. "
                         "--val-fraction erhöhen oder --knn-k 0 setzen.")
    if len(val_groups) < 3:
        print("  Hinweis: Weniger als 3 Validierungsgruppen; Schwellen und "
              "Phasenkalibrierung sind entsprechend unsicher.")

    frames_train = FrameDataset(
        inputs_dir, "trainval", train_folds, only_good=True,
        frame_stride=args.frame_stride, groups=set(train_groups),
    )
    frames_val = FrameDataset(
        inputs_dir, "trainval", train_folds, only_good=True,
        frame_stride=args.eval_frame_stride or 1, groups=set(val_groups),
    )
    frame_hw = frames_train.frame_hw()
    print(f"Frame: {frame_hw[0]}x{frame_hw[1]} (H x B) | Trainingsframes: "
          f"{len(frames_train)} | Validierungsframes: {len(frames_val)}")

    t0 = time.perf_counter()
    model = PhaseTemplateModel(args.bins, args.std_floor, args.shift_tol).fit(frames_train)
    train_time = time.perf_counter() - t0

    cfg = make_cfg(args, frame_hw, train_folds, train_groups, val_groups,
                   train_time, model.fill_info)

    print("\nBewerte Validierungsframes ...")
    val_df = score_frames(model, frames_val, cfg, label="val ")

    calibration = PhaseCalibration(args.phase_bins).fit(val_df)
    val_df["score_z"] = calibration.transform(val_df)
    if args.knn_k:
        val_df["score_knn"] = knn_scores(val_df, val_df, args.phase_bins, args.knn_k)
    thresholds, n_val_sweeps, n_val_groups = thresholds_from_validation(
        val_df, args.topk, args.threshold_quantile
    )

    model.save(model_dir / "templates.npz")
    save_json(model_dir / "config.json", cfg)
    save_json(model_dir / "calibration.json", {
        "phase": calibration.to_dict(),
        "thresholds": thresholds,
        "n_val_sweeps": n_val_sweeps,
        "n_val_groups": n_val_groups,
        "heatmap_vmax": float(1.5 * val_df["score"].max()),
    })
    val_df.to_csv(model_dir / "val_frame_scores.csv", index=False)
    print(f"\nTemplates gespeichert: {model_dir / 'templates.npz'}")
    return model_dir


def load_model(model_dir):
    model_dir = Path(model_dir)
    if not (model_dir / "templates.npz").is_file() or not (model_dir / "config.json").is_file():
        raise SystemExit(f"Modell nicht gefunden in {model_dir}. Zuerst 'build' bzw. 'cv' ausführen.")
    cfg = load_json(model_dir / "config.json")
    model = PhaseTemplateModel.load(
        model_dir / "templates.npz", cfg["bins"], cfg["std_floor"], cfg["shift_tol"]
    )
    return model, cfg


def evaluate_model(model_dir, inputs_dir, output_dir, split, folds, args,
                   allow_leakage=False, add_test_defects=False):
    output_dir = ensure_dir(output_dir)
    model, cfg = load_model(model_dir)
    cal = load_json(Path(model_dir) / "calibration.json")

    print("\n" + "=" * 70)
    print("PHASEN-TEMPLATE AUSWERTUNG")
    print("=" * 70)
    print(f"Modell: {model_dir} | Split: {split} | Folds: {folds}")

    if split == "trainval":
        if not folds:
            raise SystemExit("Für --split trainval muss --folds angegeben werden.")
        overlap = set(folds) & set(cfg["train_folds"])
        if overlap and not allow_leakage:
            raise SystemExit(
                f"Leakage: Folds {sorted(overlap)} wurden für dieses Modell benutzt. "
                "Mit --allow-leakage trotzdem auswerten (Ergebnis dann nicht aussagekräftig)."
            )
    elif folds:
        raise SystemExit("--folds ist nur bei --split trainval erlaubt.")

    stride = args.eval_frame_stride or cfg["eval_frame_stride"]
    frames = FrameDataset(
        inputs_dir, split, folds, only_good=False, frame_stride=stride,
        add_defects_from_test=add_test_defects,
    )
    if list(frames.frame_hw()) != list(cfg["frame_hw"]):
        raise SystemExit(
            f"Bildgröße der Daten {list(frames.frame_hw())} passt nicht zum Modell "
            f"{cfg['frame_hw']}."
        )
    print(f"Frames: {len(frames)}")

    heatmap_dir = ensure_dir(output_dir / "heatmaps") if args.save_maps else None
    t0 = time.perf_counter()
    frame_df = score_frames(model, frames, cfg, heatmap_dir, cal["heatmap_vmax"], label="eval ")
    total_time = time.perf_counter() - t0

    calibration = PhaseCalibration.from_dict(cal["phase"])
    frame_df["score_z"] = calibration.transform(frame_df)
    if cfg.get("knn_k", 0):
        reference = pd.read_csv(Path(model_dir) / "val_frame_scores.csv",
                                usecols=["group_id", "direction", "phase", "score", "label"],
                                dtype={"group_id": str, "direction": str},
                                float_precision="round_trip")
        frame_df["score_knn"] = knn_scores(frame_df, reference, calibration.bins, cfg["knn_k"])

    sweep_df, group_df = aggregate(frame_df, cfg["topk"])
    metrics = compute_metrics(sweep_df, group_df, cal["thresholds"])
    metrics.update({
        "split": split, "folds": folds, "model_dir": str(model_dir),
        "total_time_seconds": total_time,
        "mean_time_per_frame": total_time / max(len(frame_df), 1),
    })

    frame_df.to_csv(output_dir / "frame_scores.csv", index=False)
    sweep_df.to_csv(output_dir / "sweep_scores.csv", index=False)
    group_df.to_csv(output_dir / "group_scores.csv", index=False)
    metrics_table(metrics).to_csv(output_dir / "metrics_table.csv", index=False)
    save_json(output_dir / "metrics.json", metrics)

    print_metrics(metrics)
    print(f"\nErgebnisse: {output_dir}")
    return frame_df, metrics


def cmd_build(args):
    train_model(args.inputs, args.store, trainval_folds(args.inputs), "final", args)


def cmd_score(args):
    evaluate_model(
        Path(args.store) / args.model_name, args.inputs,
        Path(args.output) / args.model_name,
        args.split, args.folds, args, args.allow_leakage,
    )


def cmd_cv(args):
    inputs_dir = Path(args.inputs)
    store_dir = ensure_dir(args.store)
    folds = trainval_folds(inputs_dir)
    if len(folds) < 2:
        raise SystemExit("Für Cross-Validation werden mindestens zwei Folds benötigt.")

    index = pd.read_csv(inputs_dir / "index.csv")
    gcol = group_col(index)
    trainval_has_defects = bool(((index["split"] == "trainval") & (index["label"] == 1)).any())
    add_test_defects = not trainval_has_defects
    if add_test_defects:
        n_def = index.loc[(index["split"] == "test") & (index["label"] == 1), gcol].nunique()
        print(f"trainval enthält keine Fehler: jede Fold-Auswertung = guter Fold + "
              f"alle {n_def} fehlerhaften Gruppen des Testsets (nie benutzt).")

    per_fold_rows, pooled_frames = [], []
    for eval_fold in folds:
        train_folds = [f for f in folds if f != eval_fold]
        model_name = f"fold_{eval_fold}"
        print("\n" + "#" * 70)
        print(f"CV: Evaluationsfold {eval_fold} | Templates aus Folds {train_folds}")
        print("#" * 70)
        model_dir = train_model(inputs_dir, store_dir, train_folds, model_name, args)
        frame_df, metrics = evaluate_model(
            model_dir, inputs_dir, Path(args.output) / model_name,
            "trainval", [eval_fold], args, add_test_defects=add_test_defects,
        )
        pooled_frames.append(frame_df.assign(fold=eval_fold))
        for level in ("sweep", "group"):
            for combo, m in metrics[level].items():
                per_fold_rows.append({"fold": eval_fold, "level": level, "score": combo, **m})

    cv_dir = ensure_dir(Path(args.output) / "cv")
    per_fold = pd.DataFrame(per_fold_rows)
    per_fold.to_csv(cv_dir / "cv_results.csv", index=False)

    pooled = pd.concat(pooled_frames, ignore_index=True)
    pooled["bottle_id"] = pooled["fold"].astype(str) + "/" + pooled["bottle_id"]
    pooled["group_id"] = pooled["fold"].astype(str) + "/" + pooled["group_id"]
    p_sweep, p_group = aggregate(pooled, args.topk)
    pooled_metrics = compute_metrics(p_sweep, p_group, thresholds=None)

    summary = []
    for level, combos in (("sweep", SWEEP_COMBOS), ("group", GROUP_COMBOS)):
        for combo in combos:
            if combo not in pooled_metrics[level]:
                continue
            sub = per_fold[(per_fold["level"] == level) & (per_fold["score"] == combo)]
            aur = pd.to_numeric(sub["auroc"], errors="coerce").dropna()
            ap = pd.to_numeric(sub["ap"], errors="coerce").dropna()
            summary.append({
                "level": level, "score": combo,
                "auroc_mean": float(aur.mean()) if len(aur) else None,
                "auroc_std": float(aur.std()) if len(aur) > 1 else None,
                "ap_mean": float(ap.mean()) if len(ap) else None,
                "auroc_pooled_oof": pooled_metrics[level][combo]["auroc"],
                "ap_pooled_oof": pooled_metrics[level][combo]["ap"],
                "folds_with_auroc": int(len(aur)),
            })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(cv_dir / "cv_summary.csv", index=False)
    save_json(cv_dir / "cv_summary.json", {"summary": summary, "pooled_oof": pooled_metrics})

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION ABGESCHLOSSEN")
    print("=" * 70)
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"))
    print(f"\nErgebnisse: {cv_dir}")


# ============================================================
# ARGUMENTE
# ============================================================

def create_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--inputs", required=True, help="Ordner mit index.csv, stats.json und *.npy")
        p.add_argument("--store", default="D:/Baseline_Store", help="Modellordner")
        p.add_argument("--output", default="D:/Baseline_Results", help="Ergebnisordner")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--save-maps", action="store_true", help="Frame + Heatmap als PNG speichern")

    def training(p):
        p.add_argument("--bins", type=int, default=50, help="Phasen-Bins je Richtung für die Templates")
        p.add_argument("--std-floor", type=float, default=0.02,
                       help="Untergrenze der Pixel-Std (Intensität 0..1)")
        p.add_argument("--shift-tol", type=int, default=2,
                       help="Verschiebungstoleranz in Pixeln (0 = aus)")
        p.add_argument("--frame-stride", type=int, default=1, help="Nur Templates: jeder n-te Frame")
        p.add_argument("--val-fraction", type=float, default=0.1,
                       help="Anteil guter Gruppen, der nie für Templates benutzt wird")
        p.add_argument("--sigma", type=float, default=4.0, help="Gauß-Glättung der Karte")
        p.add_argument("--pixel-percent", type=float, default=0.0,
                       help="0 = Maximum der Karte; sonst Mittel der höchsten X %% Pixel")
        p.add_argument("--topk", type=int, default=5, help="Frames je Sweep für den Top-k-Score")
        p.add_argument("--knn-k", type=int, default=0,
                       help="Anzahl nächster guter Rohscores je Richtung/Phasen-Bin (0 = aus)")
        p.add_argument("--phase-bins", type=int, default=10, help="Phasen-Bins für score_z (0 = aus)")
        p.add_argument("--threshold-quantile", type=float, default=0.95)

    def evaluation(p):
        p.add_argument("--eval-frame-stride", type=int, default=None,
                       help="Auswertung/Validierung: jeder n-te Frame (Standard 1)")

    b = sub.add_parser("build", help="Schätzt die Templates (Modell final)")
    common(b)
    training(b)
    evaluation(b)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="Bewertet ein Modell")
    common(s)
    evaluation(s)
    s.add_argument("--split", choices=["trainval", "test"], default="test")
    s.add_argument("--folds", nargs="*", type=int, default=None)
    s.add_argument("--model-name", default="final")
    s.add_argument("--allow-leakage", action="store_true")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("cv", help="Cross-Validation")
    common(c)
    training(c)
    evaluation(c)
    c.set_defaults(func=cmd_cv)
    return parser


def main():
    args = create_parser().parse_args()
    np.random.seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
