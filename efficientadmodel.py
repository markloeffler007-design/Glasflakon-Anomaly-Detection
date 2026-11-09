
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EfficientAD für Glasflakon-Anomalieerkennung.

Eigenschaften:
- Anomalib 2.7.0
- EfficientAD Small
- CPU-Training
- Originalbilder: 1920 x 1080
- Überlappende 512x512-Kacheln
- Training ausschließlich mit guten Trainval-Bildern
- Separate Modelle für CV-Folds
- Unabhängiges Testset
- Frame-, Bottle- und Group-Scores
- Heatmaps in Originalauflösung

Befehle:
    build   Trainiert ein Modell mit allen guten Trainval-Daten.
    score   Bewertet einen Fold oder das unabhängige Testset.
    cv      Führt Cross-Validation durch.

Beispiel:
    python efficientadmodell.py build --inputs "C:/Glasflakons/inputs" --store "D:/EfficientAD"

    python efficientadmodell.py score --inputs "C:/Glasflakons/inputs" --store "D:/EfficientAD" --split test

    python efficientadmodell.py cv --inputs "C:/Glasflakons/inputs" --store "D:/EfficientAD"
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

from anomalib.data import ImageItem, ImageBatch
from anomalib.engine import Engine
from anomalib.models import EfficientAd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from scipy.ndimage import gaussian_filter

from bottle_dataset import BottleDataset


# ============================================================
# KONFIGURATION
# ============================================================

TILE_SIZE = 512
TILE_OVERLAP = 128

DEFAULT_STEPS = 2000
DEFAULT_LR = 0.0001
DEFAULT_WEIGHT_DECAY = 0.00001

DEVICE = "cpu"

torch.set_num_threads(8)

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# ALLGEMEINE HILFSFUNKTIONEN
# ============================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_device():
    return torch.device(DEVICE)


def tile_starts(length, tile_size, overlap):
    """
    Erzeugt Kachelpositionen.

    Die letzte Kachel wird am Bildrand verankert,
    damit kein Bildbereich unberücksichtigt bleibt.
    """

    if length <= tile_size:
        return [0]

    step = tile_size - overlap

    positions = list(range(0, length - tile_size + 1, step))

    last = length - tile_size

    if positions[-1] != last:
        positions.append(last)

    return sorted(set(positions))


def make_tiles(image):
    """
    image: Tensor (C,H,W)

    Rückgabe:
        Liste aus:
        (tile, x, y)
    """

    _, height, width = image.shape

    xs = tile_starts(width, TILE_SIZE, TILE_OVERLAP)
    ys = tile_starts(height, TILE_SIZE, TILE_OVERLAP)

    tiles = []

    for y in ys:
        for x in xs:
            tile = image[
                :,
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            if tile.shape[-2:] != (TILE_SIZE, TILE_SIZE):
                raise RuntimeError(
                    f"Ungültige Kachelgröße: {tile.shape}"
                )

            tiles.append((tile, x, y))

    return tiles


# ============================================================
# DATASET-ADAPTER
# ============================================================

class TiledBottleDataset(Dataset):
    """
    Adapter für die vorhandene BottleDataset-Klasse.

    Die BottleDataset liefert normalisierte Graustufenbilder.

    EfficientAD benötigt:
        RGB
        Wertebereich [0,1]
        Keine Dataset-Normalisierung

    Deshalb wird die Normalisierung hier rückgängig gemacht.
    """

    def __init__(
        self,
        inputs_dir,
        split="trainval",
        folds=None,
        only_good=True,
        frame_stride=4,
        only_valid=True,
    ):

        self.base = BottleDataset(
            inputs_dir=inputs_dir,
            split=split,
            folds=folds,
            only_good=only_good,
            mode="frame",
            frame_stride=frame_stride,
            only_valid=only_valid,
        )

        self.mean = self.base.mean
        self.std = self.base.std

        self.items = []

        for index in range(len(self.base)):
            sample = self.base[index]

            gray = sample["x"][0]

            gray = gray * self.std + self.mean
            gray = gray.clamp(0.0, 1.0)

            rgb = gray.unsqueeze(0).repeat(3, 1, 1)

            for tile, x, y in make_tiles(rgb):
                self.items.append({
                    "image": tile,
                    "label": sample["label"],
                    "bottle_id": sample["bottle_id"],
                    "phase": float(sample["phase"]),
                    "group_id": sample.get("group_id"),
                    "direction": sample.get("direction"),
                    "x": x,
                    "y": y,
                })

        if not self.items:
            raise RuntimeError("Keine Kacheln im Dataset erzeugt.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        item = self.items[index]

        return ImageItem(
            image=item["image"],
            gt_label=torch.tensor(
                item["label"],
                dtype=torch.int64
            ),
            image_path=f"tile_{index}.png",
        )


# ============================================================
# ANOMALIB DATAMODULE
# ============================================================

class BottleDataModule(LightningDataModule):

    def __init__(
        self,
        train_dataset,
        batch_size=1,
        num_workers=0,
    ):

        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = train_dataset

        self.train_batch_size = batch_size
        self.val_batch_size = batch_size

        self.num_workers = num_workers

    @staticmethod
    def collate_fn(items):
        return ImageBatch.collate(items)

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
        )


# ============================================================
# MODELL ERZEUGEN
# ============================================================

def create_model(args):

    model = EfficientAd(
        imagenet_dir=str(
            Path(args.store) / "imagenette"
        ),
        model_size="s",
        lr=args.lr,
        weight_decay=args.weight_decay,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(
    inputs_dir,
    store_dir,
    train_folds,
    model_name,
    args,
):

    store_dir = ensure_dir(store_dir)

    model_dir = ensure_dir(
        store_dir / model_name
    )

    print("\n" + "=" * 70)
    print("EFFICIENTAD TRAINING")
    print("=" * 70)

    print(f"Train-Folds: {train_folds}")
    print(f"Modellordner: {model_dir}")
    print(f"Trainingsschritte: {args.steps}")
    print(f"Kachelgröße: {TILE_SIZE}")
    print(f"Gerät: {DEVICE}")

    start_time = time.perf_counter()

    dataset = TiledBottleDataset(
        inputs_dir=inputs_dir,
        split="trainval",
        folds=train_folds,
        only_good=True,
        frame_stride=args.frame_stride,
    )

    print(f"Trainingskacheln: {len(dataset)}")

    datamodule = BottleDataModule(
        train_dataset=dataset,
        batch_size=1,
        num_workers=args.workers,
    )

    model = create_model(args)

    engine = Engine(
        max_steps=args.steps,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )

    print("\nStarte EfficientAD-Training ...")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    checkpoint_path = model_dir / "efficientad_state.pt"

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    config = {
        "model": "EfficientAD",
        "model_size": "s",
        "anomalib_version": "2.7.0",
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "steps": args.steps,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "frame_stride": args.frame_stride,
        "train_folds": train_folds,
        "image_channels": 3,
        "input_normalization": False,
    }

    save_json(
        model_dir / "config.json",
        config,
    )

    duration = time.perf_counter() - start_time

    print("\nTraining abgeschlossen.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Trainingszeit: {duration:.1f} Sekunden")

    return model_dir


# ============================================================
# MODELL LADEN
# ============================================================

def load_model(model_dir, args):

    model_dir = Path(model_dir)

    checkpoint = model_dir / "efficientad_state.pt"

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint nicht gefunden: {checkpoint}"
        )

    model = create_model(args)

    # Teacher-Gewichte initialisieren.
    model.prepare_pretrained_model()

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()
    model.to(get_device())

    return model


# ============================================================
# ANOMALIE-HEATMAP
# ============================================================

@torch.no_grad()
def predict_tile(model, tile):

    tile = tile.unsqueeze(0).to(get_device())

    # EfficientAD liefert zwei Anomaliekarten:
    # Student-Teacher und Student-Autoencoder.

    map_st, map_ae = model.model.get_maps(
        tile,
        normalize=True,
    )

    anomaly_map = map_st + map_ae

    anomaly_map = F.interpolate(
        anomaly_map,
        size=(TILE_SIZE, TILE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return anomaly_map[0, 0].cpu().numpy()


@torch.no_grad()
def predict_full_image(model, image):

    _, height, width = image.shape

    anomaly_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    tiles = make_tiles(image)

    for tile, x, y in tiles:

        anomaly_map = predict_tile(
            model,
            tile,
        )

        anomaly_sum[
            y:y + TILE_SIZE,
            x:x + TILE_SIZE
        ] += anomaly_map

        weight_sum[
            y:y + TILE_SIZE,
            x:x + TILE_SIZE
        ] += 1.0

    anomaly_map = anomaly_sum / np.maximum(
        weight_sum,
        1e-8,
    )

    anomaly_map = gaussian_filter(
        anomaly_map,
        sigma=2.0,
    )

    return anomaly_map


# ============================================================
# SCORE-BERECHNUNG
# ============================================================

def calculate_frame_score(anomaly_map):

    flat = anomaly_map.reshape(-1)

    # Mittelwert der obersten 1 Prozent
    # statt ausschließlich des Maximums.

    count = max(
        1,
        int(len(flat) * 0.01),
    )

    largest = np.partition(
        flat,
        len(flat) - count,
    )[-count:]

    return float(largest.mean())


def save_heatmap(anomaly_map, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    minimum = float(anomaly_map.min())
    maximum = float(anomaly_map.max())

    normalized = (
        anomaly_map - minimum
    ) / max(
        maximum - minimum,
        1e-8,
    )

    image = Image.fromarray(
        np.uint8(normalized * 255)
    )

    image.save(output_path)


# ============================================================
# METRIKEN
# ============================================================

def calculate_metrics(df, score_col="score"):

    labels = df["label"].astype(int).to_numpy()
    scores = df[score_col].astype(float).to_numpy()

    result = {
        "n_samples": int(len(df)),
        "n_good": int((labels == 0).sum()),
        "n_bad": int((labels == 1).sum()),
    }

    if len(np.unique(labels)) < 2:

        result["auroc"] = None
        result["average_precision"] = None
        result["f1"] = None

        return result

    result["auroc"] = float(
        roc_auc_score(labels, scores)
    )

    result["average_precision"] = float(
        average_precision_score(labels, scores)
    )

    # Schwellenwert ausschließlich aus den Scores
    # der guten Trainingsdaten ableiten.
    #
    # Für eine spätere endgültige Bewertung sollte
    # dieser Schwellenwert anhand einer separaten
    # Validierungsmenge festgelegt werden.

    threshold = float(
        np.quantile(
            scores[labels == 0],
            0.99,
        )
    )

    predictions = (
        scores >= threshold
    ).astype(int)

    result["threshold"] = threshold

    result["precision"] = float(
        precision_score(
            labels,
            predictions,
            zero_division=0,
        )
    )

    result["recall"] = float(
        recall_score(
            labels,
            predictions,
            zero_division=0,
        )
    )

    result["f1"] = float(
        f1_score(
            labels,
            predictions,
            zero_division=0,
        )
    )

    return result


def aggregate_scores(frame_df, group_column):

    if group_column not in frame_df.columns:
        return pd.DataFrame()

    df = (
        frame_df
        .groupby(group_column, dropna=False)
        .agg(
            score=("score", "max"),
            label=("label", "max"),
            n_frames=("score", "count"),
        )
        .reset_index()
    )

    return df


# ============================================================
# AUSWERTUNG
# ============================================================

def evaluate_model(
    model_dir,
    inputs_dir,
    output_dir,
    split,
    folds,
    args,
):

    output_dir = ensure_dir(output_dir)

    print("\n" + "=" * 70)
    print("EFFICIENTAD AUSWERTUNG")
    print("=" * 70)

    print(f"Split: {split}")
    print(f"Folds: {folds}")

    model = load_model(
        model_dir,
        args,
    )

    dataset = BottleDataset(
        inputs_dir=inputs_dir,
        split=split,
        folds=folds,
        only_good=False,
        mode="frame",
        frame_stride=args.frame_stride,
        only_valid=True,
    )

    print(f"Frames: {len(dataset)}")

    frame_rows = []

    total_start = time.perf_counter()

    heatmap_dir = ensure_dir(
        output_dir / "heatmaps"
    )

    for index in range(len(dataset)):

        sample_start = time.perf_counter()

        sample = dataset[index]

        gray = sample["x"][0]

        gray = (
            gray * dataset.std + dataset.mean
        ).clamp(0.0, 1.0)

        image = gray.unsqueeze(0).repeat(
            3,
            1,
            1,
        )

        anomaly_map = predict_full_image(
            model,
            image,
        )

        score = calculate_frame_score(
            anomaly_map
        )

        base_index, frame_index = dataset.items[index]

        row = dataset.df.iloc[base_index]

        bottle_id = str(sample["bottle_id"])

        group_id = sample.get("group_id")

        frame_rows.append({
            "frame_index": int(frame_index),
            "bottle_id": bottle_id,
            "group_id": group_id,
            "direction": sample.get("direction"),
            "phase": float(sample["phase"]),
            "label": int(sample["label"]),
            "score": score,
            "inference_time": (
                time.perf_counter() - sample_start
            ),
        })

        if args.save_maps:

            safe_name = (
                bottle_id
                .replace("/", "_")
                .replace("\\", "_")
            )

            save_heatmap(
                anomaly_map,
                heatmap_dir /
                f"{safe_name}_{frame_index:06d}.png",
            )

        if (
            index == 0
            or (index + 1) % 50 == 0
            or index + 1 == len(dataset)
        ):

            print(
                f"[{index + 1}/{len(dataset)}] "
                f"Score={score:.5f} "
                f"Zeit={time.perf_counter() - sample_start:.2f}s"
            )

    total_time = time.perf_counter() - total_start

    frame_df = pd.DataFrame(frame_rows)

    bottle_df = aggregate_scores(
        frame_df,
        "bottle_id",
    )

    group_df = aggregate_scores(
        frame_df,
        "group_id",
    )

    frame_df.to_csv(
        output_dir / "frame_scores.csv",
        index=False,
    )

    bottle_df.to_csv(
        output_dir / "bottle_scores.csv",
        index=False,
    )

    group_df.to_csv(
        output_dir / "group_scores.csv",
        index=False,
    )

    metrics = {
        "split": split,
        "folds": folds,
        "total_time_seconds": total_time,
        "mean_time_per_frame": (
            total_time / max(len(frame_df), 1)
        ),
        "frame_metrics": calculate_metrics(
            frame_df
        ),
        "bottle_metrics": calculate_metrics(
            bottle_df
        ),
        "group_metrics": (
            calculate_metrics(group_df)
            if len(group_df)
            else None
        ),
    }

    save_json(
        output_dir / "metrics.json",
        metrics,
    )

    print("\nAuswertung abgeschlossen.")

    print(
        json.dumps(
            metrics,
            indent=4,
            ensure_ascii=False,
        )
    )

    return metrics


# ============================================================
# BUILD
# ============================================================

def cmd_build(args):

    inputs_dir = Path(args.inputs)
    store_dir = ensure_dir(args.store)

    index = pd.read_csv(
        inputs_dir / "index.csv"
    )

    folds = sorted(
        index.loc[
            index["split"] == "trainval",
            "fold",
        ].dropna().unique().tolist()
    )

    train_model(
        inputs_dir=inputs_dir,
        store_dir=store_dir,
        train_folds=folds,
        model_name="final",
        args=args,
    )


# ============================================================
# SCORE
# ============================================================

def cmd_score(args):

    model_name = args.model_name

    model_dir = (
        Path(args.store) / model_name
    )

    evaluate_model(
        model_dir=model_dir,
        inputs_dir=args.inputs,
        output_dir=Path(args.output) / model_name,
        split=args.split,
        folds=args.folds,
        args=args,
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

def cmd_cv(args):

    inputs_dir = Path(args.inputs)
    store_dir = ensure_dir(args.store)

    index = pd.read_csv(
        inputs_dir / "index.csv"
    )

    folds = sorted(
        index.loc[
            index["split"] == "trainval",
            "fold",
        ].dropna().unique().tolist()
    )

    if len(folds) < 2:
        raise RuntimeError(
            "Für Cross-Validation werden mindestens zwei Folds benötigt."
        )

    all_results = []

    for eval_fold in folds:

        train_folds = [
            fold
            for fold in folds
            if fold != eval_fold
        ]

        model_name = f"fold_{eval_fold}"

        print("\n")
        print("#" * 70)
        print(f"CV: Evaluationsfold {eval_fold}")
        print(f"Training auf Folds {train_folds}")
        print("#" * 70)

        model_dir = train_model(
            inputs_dir=inputs_dir,
            store_dir=store_dir,
            train_folds=train_folds,
            model_name=model_name,
            args=args,
        )

        result_dir = (
            Path(args.output) / model_name
        )

        metrics = evaluate_model(
            model_dir=model_dir,
            inputs_dir=inputs_dir,
            output_dir=result_dir,
            split="trainval",
            folds=[eval_fold],
            args=args,
        )

        all_results.append({
            "fold": eval_fold,
            "auroc_frame": metrics[
                "frame_metrics"
            ]["auroc"],
            "auroc_bottle": metrics[
                "bottle_metrics"
            ]["auroc"],
            "auroc_group": (
                metrics["group_metrics"]["auroc"]
                if metrics["group_metrics"]
                else None
            ),
            "total_time_seconds": metrics[
                "total_time_seconds"
            ],
        })

    results_df = pd.DataFrame(
        all_results
    )

    cv_dir = ensure_dir(
        Path(args.output) / "cv"
    )

    results_df.to_csv(
        cv_dir / "cv_results.csv",
        index=False,
    )

    numeric_columns = [
        "auroc_frame",
        "auroc_bottle",
        "auroc_group",
        "total_time_seconds",
    ]

    summary = {
        column: {
            "mean": (
                float(results_df[column].mean())
                if results_df[column].notna().any()
                else None
            ),
            "std": (
                float(results_df[column].std())
                if results_df[column].notna().any()
                else None
            ),
        }
        for column in numeric_columns
    }

    save_json(
        cv_dir / "cv_summary.json",
        summary,
    )

    print("\nCross-Validation abgeschlossen.")
    print(results_df)
    print(summary)


# ============================================================
# ARGUMENTE
# ============================================================

def create_parser():

    parser = argparse.ArgumentParser(
        description="EfficientAD für Glasflakons"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    def add_common_arguments(p):

        p.add_argument(
            "--inputs",
            required=True,
            help="Ordner mit inputs und index.csv",
        )

        p.add_argument(
            "--store",
            default="D:/EfficientAD_Store",
        )

        p.add_argument(
            "--output",
            default="D:/EfficientAD_Results",
        )

        p.add_argument(
            "--steps",
            type=int,
            default=DEFAULT_STEPS,
        )

        p.add_argument(
            "--lr",
            type=float,
            default=DEFAULT_LR,
        )

        p.add_argument(
            "--weight-decay",
            type=float,
            default=DEFAULT_WEIGHT_DECAY,
        )

        p.add_argument(
            "--frame-stride",
            type=int,
            default=4,
        )

        p.add_argument(
            "--workers",
            type=int,
            default=0,
        )

        p.add_argument(
            "--save-maps",
            action="store_true",
        )

    # BUILD
    build_parser = subparsers.add_parser(
        "build",
        help="Trainiert das finale Modell",
    )

    add_common_arguments(build_parser)

    build_parser.set_defaults(
        func=cmd_build
    )

    # SCORE
    score_parser = subparsers.add_parser(
        "score",
        help="Bewertet ein Modell",
    )

    add_common_arguments(score_parser)

    score_parser.add_argument(
        "--split",
        choices=["trainval", "test"],
        default="test",
    )

    score_parser.add_argument(
        "--folds",
        nargs="*",
        type=int,
        default=None,
    )

    score_parser.add_argument(
        "--model-name",
        default="final",
    )

    score_parser.set_defaults(
        func=cmd_score
    )

    # CV
    cv_parser = subparsers.add_parser(
        "cv",
        help="Führt Cross-Validation durch",
    )

    add_common_arguments(cv_parser)

    cv_parser.set_defaults(
        func=cmd_cv
    )

    return parser


# ============================================================
# MAIN
# ============================================================

def main():

    parser = create_parser()

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    args.func(args)


if __name__ == "__main__":
    main()
