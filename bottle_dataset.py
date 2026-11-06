"""
BottleDataset für die direkt von Preprocessing.py erzeugte inputs-Struktur.

Kompatibel mit patchcoremodell.py:
- split: trainval oder test
- folds: Auswahl der trainval-Folds
- only_good: nur Label 0
- mode="frame": einzelner Frame, (1,H,W)
- mode="sequence": Framefenster, (1,T,H,W)

Es findet keine zeitliche Synchronisation statt. lag ist im Index standardmäßig 0.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BottleDataset(Dataset):
    def __init__(
        self, inputs_dir, split="trainval", folds=None, only_good=False,
        mode="sequence", window=16, stride=8, frame_stride=4, only_valid=True
    ):
        self.dir = Path(inputs_dir)
        index_path = self.dir / "index.csv"
        stats_path = self.dir / "stats.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"index.csv fehlt: {index_path}")
        if not stats_path.is_file():
            raise FileNotFoundError(f"stats.json fehlt: {stats_path}")

        df = pd.read_csv(index_path)
        required = {"split", "fold", "label", "valid_start", "valid_end",
                    "n_frames", "lag", "path", "bottle_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"index.csv fehlen Spalten: {sorted(missing)}")

        df = df[df["split"] == split]
        if folds is not None:
            df = df[df["fold"].isin(folds)]
        if only_good:
            df = df[df["label"] == 0]
        if df.empty:
            raise ValueError(
                f"Keine Sequenzen für split={split}, folds={folds}, only_good={only_good}"
            )

        self.df = df.reset_index(drop=True)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.mean = float(stats["mean"])
        self.std = max(float(stats["std"]), 1e-8)
        self.mode = mode
        self.window = int(window)
        self.stride = int(stride)
        self.frame_stride = int(frame_stride)

        if mode not in {"sequence", "frame"}:
            raise ValueError("mode muss 'sequence' oder 'frame' sein.")
        if self.window < 1 or self.stride < 1 or self.frame_stride < 1:
            raise ValueError("window und stride müssen positiv sein.")

        self.items = []
        for i, row in self.df.iterrows():
            lo = int(row.valid_start) if only_valid else 0
            hi = int(row.valid_end) if only_valid else int(row.n_frames)
            if mode == "sequence":
                starts = range(lo, max(lo, hi - self.window + 1), self.stride)
            else:
                starts = range(lo, hi, self.frame_stride)
            self.items.extend((i, int(t)) for t in starts)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, t = self.items[k]
        row = self.df.iloc[i]
        path = self.dir / str(row.path)
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError(f"{path}: erwartet (T,H,W), erhalten {arr.shape}")

        lag = int(row.lag) if "lag" in row.index else 0
        total = arr.shape[0]
        if total == 0:
            raise ValueError(f"Leere Sequenz: {path}")

        if self.mode == "sequence":
            start = t + lag
            indices = np.clip(np.arange(start, start + self.window), 0, total - 1)
            x = np.asarray(arr[indices])[None, ...]  # (1,T,H,W)
        else:
            idx = int(np.clip(t + lag, 0, total - 1))
            x = np.asarray(arr[idx])[None, ...]      # (1,H,W)

        x = torch.from_numpy(np.array(x, copy=True)).float() / 255.0
        x = (x - self.mean) / self.std
        phase = float(t / max(int(row.n_frames), 1))

        result = {
            "x": x,
            "label": int(row.label),
            "bottle_id": str(row.bottle_id),
            "phase": torch.tensor(phase, dtype=torch.float32),
        }
        if "group_id" in row.index:
            result["group_id"] = str(row.group_id)
        if "direction" in row.index:
            result["direction"] = str(row.direction)
        return result
