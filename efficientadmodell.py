#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientAD (Anomalib 2.7.0) für die Fehlererkennung an Glasflakons.

Eingabe: die von Preprocessing.py erzeugte inputs-Struktur (index.csv, stats.json,
*.npy mit Form (T, H, W)), gelesen über bottle_dataset.BottleDataset.

Design (auf die Aufgabe zugeschnitten)
- Unüberwachtes Training ausschließlich mit guten Frames aus trainval.
- Ganzer Frame als Modelleingabe (Standard). Die Bildgröße kommt aus dem Preprocessing
  (z. B. 270 x 480). Kacheln sind optional (--tile-size), falls später deutlich
  größere Bilder verwendet werden.
- Ein Teil der guten trainval-GRUPPEN wird nie trainiert, sondern dient als
  Validierung: Quantile der Anomaliekarten (EfficientAD-Normierung), Kalibrierung
  der Scores je Scanrichtung und Phase sowie Entscheidungsschwellen.
  Die Aufteilung erfolgt nach group_id, damit Hin- und Rückfahrt einer Messung
  nie getrennt werden.
- Auswertung auf jedem Frame (--eval-frame-stride 1), dann Aggregation
  Frame -> Sweep (bottle_id = eine Fahrtrichtung) -> Gruppe (group_id = Messung).
- Zwei Score-Arten: "score" (roh) und "score_z" (phasenkalibriert: je Richtung und
  Phasen-Bin robust standardisiert mit den Statistiken der guten Validierungsframes).
- AUROC und Average Precision sind schwellenfrei und die Hauptkennzahlen.
  Precision/Recall/F1 beziehen sich auf Schwellen, die nur aus guten Validierungs-
  daten stammen (nie aus den ausgewerteten Daten).
- Cross-Validation mit gepoolten Out-of-Fold-Metriken, analog zu patchcoremodell.py.

Befehle
    build   Trainiert ein Modell mit allen guten trainval-Folds (Modellname "final").
    score   Bewertet ein Modell auf dem Testset oder auf nicht trainierten Folds.
    cv      K-Fold-Cross-Validation (je Fold ein eigenes Modell).
    predict Bewertet neue Roh-Bilder (Datei oder Ordner) mit einem trainierten Modell.

Beispiele
    python efficientadmodell.py build --inputs "C:/Glasflakons/Input" --store "D:/EfficientAD_Store"
    python efficientadmodell.py score --inputs "C:/Glasflakons/Input" --store "D:/EfficientAD_Store" --split test
    python efficientadmodell.py cv    --inputs "C:/Glasflakons/Input" --store "D:/EfficientAD_Store"

Hinweis zu Downloads: Beim ersten Training lädt Anomalib die vortrainierten
Teacher-Gewichte (GitHub) und Imagenette (ca. 1,5 GB, nach <store>/imagenette).
Imagenette dient nur dem Penalty-Term des Student-Verlusts.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningDataModule, seed_everything

import anomalib
from anomalib.callbacks import ModelCheckpoint
from anomalib.data import ImageBatch, ImageItem
from anomalib.engine import Engine
from anomalib.models import EfficientAd

from bottle_dataset import BottleDataset


# ============================================================
# KONFIGURATION
# ============================================================

# Der Autoencoder von EfficientAD braucht pro Bildseite mindestens 256 px
# (enconv6 hat Kernel 8 auf einer Feature-Map der Größe H/32).
MIN_SIDE = 256

DEFAULT_STEPS = 5000
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5

# Quantile für die Normierung der Anomaliekarten (wie Anomalib / EfficientAD-Paper).
QUANTILE_LOW = 0.90
QUANTILE_HIGH = 0.995

# Untergrenze der Skala je Phasenzelle relativ zur globalen Skala, damit sehr ruhige
# Zellen die z-Scores nicht explodieren lassen.
MIN_SCALE_FRACTION = 0.25

SCORE_COLS = ("score", "score_z")
SWEEP_COMBOS = [f"{c}_{a}" for c in SCORE_COLS for a in ("max", "topk")]
GROUP_COMBOS = [f"{s}_{d}" for s in SWEEP_COMBOS for d in ("dirmean", "dirmax")]

warnings.filterwarnings("ignore", message=".*does not have many workers.*")


# ============================================================
# ALLGEMEINE HILFSFUNKTIONEN
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


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda gewählt, aber keine CUDA-GPU verfügbar.")
    return torch.device(name)


def group_col(df):
    for col in ("group_id", "group"):
        if col in df.columns:
            return col
    return "bottle_id"


def tile_starts(length, tile, overlap):
    """Kachelpositionen; die letzte Kachel wird am Bildrand verankert."""
    if length <= tile:
        return [0]
    step = tile - overlap
    positions = list(range(0, length - tile + 1, step))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def tile_grid(height, width, tile, overlap):
    return [
        (y, x)
        for y in tile_starts(height, tile, overlap)
        for x in tile_starts(width, tile, overlap)
    ]


# ============================================================
# DATEN
# ============================================================

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


class ModelUnits(Dataset):
    """
    Modelleinheiten als ImageItem (3 Kanäle, [0, 1]): ganzer Frame oder Kacheln.
    Wird für Training und für die Quantilberechnung verwendet.
    """

    def __init__(self, frames, tile_size=0, tile_overlap=0):
        self.frames = frames
        h, w = frames.frame_hw()
        self.tile_size = int(tile_size)
        self.tiles = (
            tile_grid(h, w, self.tile_size, tile_overlap)
            if self.tile_size > 0 else [None]
        )
        self.index = [
            (k, t) for k in range(len(frames)) for t in range(len(self.tiles))
        ]

    def unit_hw(self):
        if self.tile_size > 0:
            return (self.tile_size, self.tile_size)
        return self.frames.frame_hw()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        k, t = self.index[i]
        sample = self.frames[k]
        gray = sample["gray"]
        if self.tiles[t] is not None:
            y, x = self.tiles[t]
            gray = gray[y:y + self.tile_size, x:x + self.tile_size]
        return ImageItem(
            image=gray.unsqueeze(0).repeat(3, 1, 1),
            gt_label=torch.tensor(sample["label"], dtype=torch.int64),
            image_path=f"{sample['bottle_id']}:{sample['frame_index']}:{t}",
        )


class GlasflakonDataModule(LightningDataModule):
    """
    Minimales DataModule für Engine.fit.

    - name / category werden von Engine._setup_workspace gelesen.
    - train_batch_size muss 1 sein (EfficientAd prüft das in on_train_start).
    - Es gibt bewusst keine Validierungs-Dataloader: Die Quantile werden nach dem
      Training explizit auf held-out guten Gruppen berechnet (siehe train_model).
    """

    name = "glasflakon"
    category = ""

    def __init__(self, train_dataset, num_workers=0, seed=42):
        super().__init__()
        self.train_dataset = train_dataset
        self.train_batch_size = 1
        self.num_workers = num_workers
        self.seed = seed

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            generator=generator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=ImageBatch.collate,
            pin_memory=False,
        )


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


# ============================================================
# MODELL
# ============================================================

def create_model(store_dir, model_size, lr, weight_decay):
    # Anomalib 2.x erwartet "small" oder "medium" (nicht "s" / "m").
    return EfficientAd(
        imagenet_dir=str(Path(store_dir) / "imagenette"),
        model_size=model_size,
        lr=lr,
        weight_decay=weight_decay,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )


def preflight(model, unit_hw):
    """Prüft Eingabegröße und Formen vor dem Training (inkl. Autoencoder)."""
    h, w = unit_hw
    if min(h, w) < MIN_SIDE:
        raise SystemExit(
            f"Modelleingabe {h}x{w} ist zu klein: EfficientAD braucht pro Seite "
            f"mindestens {MIN_SIDE} px. Preprocessing.py mit größerem --out-size "
            f"ausführen (aktuell z. B. 270 x 480 Breite x Höhe)."
        )
    model.eval()
    try:
        with torch.no_grad():
            model.model.get_maps(torch.zeros(1, 3, h, w), normalize=False)
    except RuntimeError as exc:
        raise SystemExit(
            f"Modelleingabe {h}x{w} ist mit EfficientAD nicht kompatibel "
            f"(Form-Konflikt Student/Autoencoder): {exc}"
        )
    model.train()


@torch.no_grad()
def predict_map(model, gray, tile_size, tile_overlap, device):
    """
    Anomaliekarte (H, W) für einen Frame.
    0.5 * (Student-Teacher-Karte + Student-Autoencoder-Karte), beide mit den
    Validierungsquantilen normiert (wie EfficientAdModel.forward).
    """
    img = gray.unsqueeze(0).repeat(3, 1, 1)
    h, w = img.shape[-2:]

    if tile_size <= 0:
        map_st, map_ae = model.model.get_maps(img.unsqueeze(0).to(device), normalize=True)
        return (0.5 * (map_st + map_ae))[0, 0].cpu().numpy()

    grid = tile_grid(h, w, tile_size, tile_overlap)
    tiles = torch.stack([img[:, y:y + tile_size, x:x + tile_size] for y, x in grid])
    map_st, map_ae = model.model.get_maps(tiles.to(device), normalize=True)
    maps = (0.5 * (map_st + map_ae))[:, 0].cpu().numpy()

    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    for (y, x), m in zip(grid, maps):
        acc[y:y + tile_size, x:x + tile_size] += m
        cnt[y:y + tile_size, x:x + tile_size] += 1.0
    return acc / np.maximum(cnt, 1.0)


def frame_score(amap, pixel_percent):
    if pixel_percent <= 0:
        return float(amap.max())
    flat = amap.reshape(-1)
    n = max(1, int(round(flat.size * pixel_percent / 100.0)))
    return float(np.partition(flat, flat.size - n)[-n:].mean())


@torch.no_grad()
def compute_map_quantiles(model, units, device, seed, max_total=2 ** 22):
    """
    Quantile (90 % / 99,5 %) der unnormierten Student-Teacher- und
    Student-Autoencoder-Karten auf held-out GUTEN Daten.

    Pro Einheit wird eine Pixelstichprobe gezogen, damit der Speicher begrenzt
    bleibt und torch.quantile (max. 2**24 Elemente) nicht überläuft.
    """
    generator = torch.Generator().manual_seed(seed)
    per_unit = max(1000, int(max_total // max(len(units), 1)))
    st_samples, ae_samples = [], []

    for i in range(len(units)):
        image = units[i].image.unsqueeze(0).to(device)
        map_st, map_ae = model.model.get_maps(image, normalize=False)
        for maps, bucket in ((map_st, st_samples), (map_ae, ae_samples)):
            flat = maps.flatten().cpu()
            n = min(per_unit, flat.numel())
            idx = torch.randperm(flat.numel(), generator=generator)[:n]
            bucket.append(flat[idx])

    result = {}
    for tag, samples in (("st", st_samples), ("ae", ae_samples)):
        flat = torch.cat(samples)
        result[f"qa_{tag}"] = torch.quantile(flat, QUANTILE_LOW).to(device)
        result[f"qb_{tag}"] = torch.quantile(flat, QUANTILE_HIGH).to(device)
    return result


# ============================================================
# PHASENKALIBRIERUNG
# ============================================================

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


# ============================================================
# SCORING, AGGREGATION, METRIKEN
# ============================================================

def save_heatmap(gray, amap, vmax, path):
    """Speichert Frame und Anomaliekarte nebeneinander (feste Skala 0..vmax)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_u8 = (gray.numpy() * 255.0).astype(np.uint8)
    heat_u8 = (np.clip(amap / max(vmax, 1e-8), 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(np.hstack([frame_u8, heat_u8])).save(path)


@torch.no_grad()
def score_frames(model, frames, cfg, device, heatmap_dir=None, heatmap_vmax=None, label=""):
    """Bewertet jeden Frame; Rückgabe: DataFrame mit Roh-Scores."""
    rows = []
    total = len(frames)
    start = time.perf_counter()

    for k in range(total):
        t0 = time.perf_counter()
        sample = frames[k]
        amap = predict_map(
            model, sample["gray"], cfg["tile_size"], cfg["tile_overlap"], device
        )
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

        if k == 0 or (k + 1) % 100 == 0 or k + 1 == total:
            print(
                f"  {label}[{k + 1}/{total}] Score={score:.4f} "
                f"({time.perf_counter() - start:.0f} s gesamt)"
            )

    return pd.DataFrame(rows)


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
            thresholds[level][combo] = float(np.quantile(df[combo].to_numpy(), quantile))
    return thresholds, len(sweep_df), len(group_df)


# ============================================================
# TRAINING
# ============================================================

def train_model(inputs_dir, store_dir, train_folds, model_name, args, device):
    store_dir = ensure_dir(store_dir)
    model_dir = ensure_dir(store_dir / model_name)

    print("\n" + "=" * 70)
    print(f"EFFICIENTAD TRAINING  ({model_name})")
    print("=" * 70)

    good_groups = list_good_groups(inputs_dir, train_folds)
    train_groups, val_groups = split_train_val_groups(
        good_groups, args.val_fraction, args.seed
    )
    print(f"Train-Folds: {train_folds}")
    print(f"Gute Gruppen: {len(good_groups)} -> Training {len(train_groups)}, "
          f"Validierung {len(val_groups)}")
    if len(val_groups) < 3:
        print("  Hinweis: Weniger als 3 Validierungsgruppen; Schwellen und "
              "Phasenkalibrierung sind entsprechend unsicher.")

    frames_train = FrameDataset(
        inputs_dir, "trainval", train_folds, only_good=True,
        frame_stride=args.frame_stride, groups=set(train_groups),
    )
    units_train = ModelUnits(frames_train, args.tile_size, args.tile_overlap)
    frames_val = FrameDataset(
        inputs_dir, "trainval", train_folds, only_good=True,
        frame_stride=args.eval_frame_stride or 1, groups=set(val_groups),
    )
    units_val = ModelUnits(frames_val, args.tile_size, args.tile_overlap)

    frame_hw = frames_train.frame_hw()
    unit_hw = units_train.unit_hw()
    print(f"Frame: {frame_hw[0]}x{frame_hw[1]} (H x B) | Modelleinheit: "
          f"{unit_hw[0]}x{unit_hw[1]} | Trainingseinheiten: {len(units_train)} | "
          f"Validierungseinheiten: {len(units_val)}")
    print(f"Schritte: {args.steps} | Modellgröße: {args.model_size} | Gerät: {device}")

    model = create_model(store_dir, args.model_size, args.lr, args.weight_decay)
    preflight(model, unit_hw)

    t_start = time.perf_counter()

    datamodule = GlasflakonDataModule(units_train, args.workers, args.seed)
    # Anomalibs Engine fügt immer einen ModelCheckpoint hinzu und verträgt sich nicht
    # mit enable_checkpointing=False. Ein eigener Callback mit save_top_k=0 erfüllt
    # die Prüfung, speichert aber nichts (das eigene state_dict folgt unten).
    engine = Engine(
        callbacks=[ModelCheckpoint(save_top_k=0)],
        default_root_dir=str(model_dir / "lightning"),
        max_steps=args.steps,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,   # Quantile werden unten explizit berechnet
    )
    print("\nStarte EfficientAD-Training ...")
    engine.fit(model=model, datamodule=datamodule)

    train_time = time.perf_counter() - t_start
    model.to(device).eval()

    # --- Quantile auf held-out guten Gruppen -------------------------------
    print("\nBerechne Normierungsquantile auf guten Validierungsdaten ...")
    quantiles = compute_map_quantiles(model, units_val, device, args.seed)
    model.model.quantiles.update(quantiles)
    if not model.model.is_set(model.model.quantiles):
        raise RuntimeError("Quantile wurden nicht gesetzt.")

    # --- Validierungsscores, Phasenkalibrierung, Schwellen -----------------
    cfg = {
        "model": "EfficientAD",
        "model_size": args.model_size,
        "anomalib_version": getattr(anomalib, "__version__", "unbekannt"),
        "steps": args.steps,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "frame_stride": args.frame_stride,
        "eval_frame_stride": args.eval_frame_stride or 1,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "sigma": args.sigma,
        "pixel_percent": args.pixel_percent,
        "topk": args.topk,
        "threshold_quantile": args.threshold_quantile,
        "frame_hw": list(frame_hw),
        "train_folds": [int(f) for f in train_folds],
        "train_groups": train_groups,
        "val_groups": val_groups,
        "seed": args.seed,
        "train_time_seconds": train_time,
    }

    print("\nBewerte Validierungsframes ...")
    val_df = score_frames(model, frames_val, cfg, device, label="val ")

    calibration = PhaseCalibration(args.phase_bins).fit(val_df)
    val_df["score_z"] = calibration.transform(val_df)
    thresholds, n_val_sweeps, n_val_groups = thresholds_from_validation(
        val_df, args.topk, args.threshold_quantile
    )

    checkpoint = model_dir / "efficientad_state.pt"
    torch.save(model.state_dict(), checkpoint)
    save_json(model_dir / "config.json", cfg)
    save_json(model_dir / "calibration.json", {
        "phase": calibration.to_dict(),
        "thresholds": thresholds,
        "n_val_sweeps": n_val_sweeps,
        "n_val_groups": n_val_groups,
        "heatmap_vmax": float(1.5 * val_df["score"].max()),
    })
    val_df.to_csv(model_dir / "val_frame_scores.csv", index=False)

    print(f"\nTraining abgeschlossen: {checkpoint}")
    print(f"Trainingszeit: {train_time:.1f} s")
    return model_dir


def load_model(model_dir, store_dir, device):
    model_dir = Path(model_dir)
    checkpoint = model_dir / "efficientad_state.pt"
    config_path = model_dir / "config.json"
    if not checkpoint.is_file() or not config_path.is_file():
        raise SystemExit(f"Modell nicht gefunden in {model_dir}. Zuerst 'build' bzw. 'cv' ausführen.")

    cfg = load_json(config_path)
    model = create_model(store_dir, cfg["model_size"], cfg["learning_rate"], cfg["weight_decay"])

    # Das state_dict enthält Teacher, Student, Autoencoder, mean_std und Quantile;
    # ein Download der Teacher-Gewichte ist hier nicht nötig.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cfg


# ============================================================
# AUSWERTUNG
# ============================================================

def evaluate_model(model_dir, inputs_dir, output_dir, split, folds, args, device,
                   allow_leakage=False, add_test_defects=False):
    output_dir = ensure_dir(output_dir)
    model, cfg = load_model(model_dir, args.store, device)
    cal = load_json(Path(model_dir) / "calibration.json")

    print("\n" + "=" * 70)
    print("EFFICIENTAD AUSWERTUNG")
    print("=" * 70)
    print(f"Modell: {model_dir} | Split: {split} | Folds: {folds}")

    if split == "trainval":
        if not folds:
            raise SystemExit("Für --split trainval muss --folds angegeben werden.")
        overlap = set(folds) & set(cfg["train_folds"])
        if overlap and not allow_leakage:
            raise SystemExit(
                f"Leakage: Folds {sorted(overlap)} wurden für dieses Modell trainiert. "
                "Mit --allow-leakage trotzdem auswerten (Ergebnis dann nicht aussagekräftig)."
            )
    elif folds:
        raise SystemExit("--folds ist nur bei --split trainval erlaubt.")

    stride = args.eval_frame_stride or cfg["eval_frame_stride"]
    if stride != cfg["eval_frame_stride"]:
        print(f"Hinweis: eval-frame-stride {stride} weicht vom Training ({cfg['eval_frame_stride']}) ab.")

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
    frame_df = score_frames(
        model, frames, cfg, device, heatmap_dir, cal["heatmap_vmax"], label="eval "
    )
    total_time = time.perf_counter() - t0

    calibration = PhaseCalibration.from_dict(cal["phase"])
    frame_df["score_z"] = calibration.transform(frame_df)

    sweep_df, group_df = aggregate(frame_df, cfg["topk"])
    metrics = compute_metrics(sweep_df, group_df, cal["thresholds"])
    metrics.update({
        "split": split,
        "folds": folds,
        "model_dir": str(model_dir),
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


# ============================================================
# BEFEHLE
# ============================================================

def trainval_folds(inputs_dir):
    index = pd.read_csv(Path(inputs_dir) / "index.csv")
    folds = index.loc[index["split"] == "trainval", "fold"].dropna().unique().tolist()
    return sorted(int(f) for f in folds)


def cmd_build(args):
    device = resolve_device(args.device)
    folds = trainval_folds(args.inputs)
    train_model(args.inputs, args.store, folds, "final", args, device)


def cmd_score(args):
    device = resolve_device(args.device)
    evaluate_model(
        Path(args.store) / args.model_name, args.inputs,
        Path(args.output) / args.model_name,
        args.split, args.folds, args, device, args.allow_leakage,
    )


def cmd_cv(args):
    device = resolve_device(args.device)
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
              f"alle {n_def} fehlerhaften Gruppen des Testsets (nie trainiert).")

    per_fold_rows = []
    pooled_frames = []
    topk = args.topk

    for eval_fold in folds:
        train_folds = [f for f in folds if f != eval_fold]
        model_name = f"fold_{eval_fold}"
        print("\n" + "#" * 70)
        print(f"CV: Evaluationsfold {eval_fold} | Training auf Folds {train_folds}")
        print("#" * 70)

        model_dir = train_model(inputs_dir, store_dir, train_folds, model_name, args, device)
        frame_df, metrics = evaluate_model(
            model_dir, inputs_dir, Path(args.output) / model_name,
            "trainval", [eval_fold], args, device,
            add_test_defects=add_test_defects,
        )
        frame_df = frame_df.assign(fold=eval_fold)
        pooled_frames.append(frame_df)
        for level in ("sweep", "group"):
            for combo, m in metrics[level].items():
                per_fold_rows.append({
                    "fold": eval_fold, "level": level, "score": combo, **m,
                })

    cv_dir = ensure_dir(Path(args.output) / "cv")
    per_fold = pd.DataFrame(per_fold_rows)
    per_fold.to_csv(cv_dir / "cv_results.csv", index=False)

    # Gepoolte Out-of-Fold-Metriken. Die Scores stammen aus verschiedenen Modellen;
    # score_z ist zwischen Folds besser vergleichbar als der Roh-Score.
    pooled = pd.concat(pooled_frames, ignore_index=True)
    pooled["bottle_id"] = pooled["fold"].astype(str) + "/" + pooled["bottle_id"]
    pooled["group_id"] = pooled["fold"].astype(str) + "/" + pooled["group_id"]
    p_sweep, p_group = aggregate(pooled, topk)
    pooled_metrics = compute_metrics(p_sweep, p_group, thresholds=None)
    p_sweep.to_csv(cv_dir / "oof_sweep_scores.csv", index=False)
    p_group.to_csv(cv_dir / "oof_group_scores.csv", index=False)

    summary = []
    for level, combos in (("sweep", SWEEP_COMBOS), ("group", GROUP_COMBOS)):
        for combo in combos:
            sub = per_fold[(per_fold["level"] == level) & (per_fold["score"] == combo)]
            aur = pd.to_numeric(sub["auroc"], errors="coerce").dropna()
            ap = pd.to_numeric(sub["ap"], errors="coerce").dropna()
            summary.append({
                "level": level,
                "score": combo,
                "auroc_mean": float(aur.mean()) if len(aur) else None,
                "auroc_std": float(aur.std()) if len(aur) > 1 else None,
                "ap_mean": float(ap.mean()) if len(ap) else None,
                "auroc_pooled_oof": pooled_metrics[level][combo]["auroc"],
                "ap_pooled_oof": pooled_metrics[level][combo]["ap"],
                "folds_with_auroc": int(len(aur)),
            })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(cv_dir / "cv_summary.csv", index=False)
    save_json(cv_dir / "cv_summary.json", {
        "summary": summary, "pooled_oof": pooled_metrics,
    })

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION ABGESCHLOSSEN")
    print("=" * 70)
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"))
    print(f"\nErgebnisse: {cv_dir}")


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_new_image(path, pre_cfg):
    """Wendet exakt die Vorverarbeitung von Preprocessing.py auf ein Rohbild an."""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Bild kann nicht gelesen werden: {path}")

    if pre_cfg.get("channel", "red") == "red":
        image = image[:, :, 2]          # OpenCV: BGR, Index 2 = Rot
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    roi = pre_cfg.get("roi")
    if roi:
        y0, y1, x0, x1 = roi
        image = image[y0:y1, x0:x1]

    width, height = pre_cfg["out_size_width_height"]
    image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image).float() / 255.0


def cmd_predict(args):
    device = resolve_device(args.device)
    model_dir = Path(args.store) / args.model_name
    model, cfg = load_model(model_dir, args.store, device)
    cal = load_json(model_dir / "calibration.json")

    pre_path = Path(args.inputs) / "config.json"
    if not pre_path.is_file():
        raise SystemExit(f"{pre_path} fehlt (wird von Preprocessing.py erzeugt).")
    pre_cfg = load_json(pre_path)

    target = Path(args.images)
    if target.is_dir():
        paths = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    elif target.is_file():
        paths = [target]
    else:
        raise SystemExit(f"Pfad nicht gefunden: {target}")
    if not paths:
        raise SystemExit(f"Keine Bilder in {target} gefunden.")

    val = pd.read_csv(model_dir / "val_frame_scores.csv")
    frame_thr = float(np.quantile(val["score"], args.frame_quantile))

    use_z = args.phase is not None and args.direction is not None
    calibration = PhaseCalibration.from_dict(cal["phase"])
    if use_z:
        frame_thr_z = float(np.quantile(val["score_z"], args.frame_quantile))

    out_dir = ensure_dir(Path(args.output) / "predict")
    rows = []
    for path in paths:
        gray = load_new_image(path, pre_cfg)
        if list(gray.shape) != list(cfg["frame_hw"]):
            raise SystemExit(
                f"Bildgröße nach Vorverarbeitung {list(gray.shape)} passt nicht zum Modell "
                f"{cfg['frame_hw']}. Passt inputs/config.json zum Training?"
            )
        amap = predict_map(model, gray, cfg["tile_size"], cfg["tile_overlap"], device)
        if cfg["sigma"] > 0:
            amap = gaussian_filter(amap, sigma=cfg["sigma"])
        score = frame_score(amap, cfg["pixel_percent"])
        row = {"bild": path.name, "score": score, "schwelle": frame_thr,
               "urteil": "DEFEKT" if score > frame_thr else "gut"}
        if use_z:
            z = float(calibration.transform(pd.DataFrame(
                {"direction": [args.direction], "phase": [args.phase], "score": [score]}))[0])
            row.update({"score_z": z, "schwelle_z": frame_thr_z,
                        "urteil_z": "DEFEKT" if z > frame_thr_z else "gut"})
        rows.append(row)
        if args.save_maps:
            save_heatmap(gray, amap, cal["heatmap_vmax"], out_dir / f"{path.stem}_heatmap.png")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if len(df) > 1:
        v = np.sort(df["score"].to_numpy())[::-1]
        s_max, s_topk = float(v[0]), float(v[:cfg["topk"]].mean())
        thr = cal["thresholds"]["sweep"]
        print(f"\nSweep-Urteil (alle {len(df)} Bilder als eine Fahrt):")
        print(f"  Max-Score  {s_max:.4f} (Schwelle {thr['score_max']:.4f}) -> "
              f"{'DEFEKT' if s_max > thr['score_max'] else 'gut'}")
        print(f"  Top-{cfg['topk']}-Score {s_topk:.4f} (Schwelle {thr['score_topk']:.4f}) -> "
              f"{'DEFEKT' if s_topk > thr['score_topk'] else 'gut'}")
    print(f"\nErgebnisse: {out_dir}")


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
        p.add_argument("--store", default="D:/EfficientAD_Store", help="Modellordner")
        p.add_argument("--output", default="D:/EfficientAD_Results", help="Ergebnisordner")
        p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        p.add_argument("--threads", type=int, default=0, help="CPU-Threads (0 = Torch-Standard)")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--save-maps", action="store_true", help="Frame + Heatmap als PNG speichern")

    def training(p):
        p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        p.add_argument("--lr", type=float, default=DEFAULT_LR)
        p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
        p.add_argument("--model-size", choices=["small", "medium"], default="small")
        p.add_argument("--frame-stride", type=int, default=2,
                       help="Nur Training: jeder n-te Frame")
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--val-fraction", type=float, default=0.1,
                       help="Anteil guter Gruppen, der nie trainiert wird (Quantile, Kalibrierung, Schwellen)")
        p.add_argument("--tile-size", type=int, default=0,
                       help="0 = ganzer Frame; sonst quadratische Kacheln (>= 256)")
        p.add_argument("--tile-overlap", type=int, default=64)
        p.add_argument("--sigma", type=float, default=4.0, help="Gauß-Glättung der Anomaliekarte")
        p.add_argument("--pixel-percent", type=float, default=0.0,
                       help="0 = Maximum der Karte; sonst Mittel der höchsten X %% Pixel")
        p.add_argument("--topk", type=int, default=5, help="Frames je Sweep für den Top-k-Score")
        p.add_argument("--phase-bins", type=int, default=10,
                       help="Phasen-Bins für score_z (0 = aus)")
        p.add_argument("--threshold-quantile", type=float, default=0.95,
                       help="Quantil der guten Validierungsscores als Entscheidungsschwelle")

    def evaluation(p):
        p.add_argument("--eval-frame-stride", type=int, default=None,
                       help="Auswertung/Validierung: jeder n-te Frame (Standard: 1; bei score der Trainingswert)")

    b = sub.add_parser("build", help="Trainiert das finale Modell")
    common(b)
    training(b)
    evaluation(b)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="Bewertet ein trainiertes Modell")
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

    pr = sub.add_parser("predict", help="Neue Roh-Bilder bewerten")
    common(pr)
    pr.add_argument("--images", required=True, help="Bilddatei oder Ordner mit Bildern (eine Fahrt)")
    pr.add_argument("--model-name", default="final")
    pr.add_argument("--frame-quantile", type=float, default=0.99,
                    help="Quantil der guten Validierungsframes als Frame-Schwelle")
    pr.add_argument("--phase", type=float, default=None,
                    help="Optional: Phase 0..1 des Bildes in der Fahrt (für score_z)")
    pr.add_argument("--direction", default=None,
                    help="Optional: Richtung wie im index.csv (z. B. forward/backward)")
    pr.set_defaults(func=cmd_predict)

    return parser


def main():
    args = create_parser().parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    seed_everything(args.seed, workers=True)
    args.func(args)


if __name__ == "__main__":
    main()
