"""Vergleich guter Flakons. Aufrufbeispiele über --help."""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def score_profiles(source, out):
    data = pd.read_csv(source)
    good = data.loc[data["label"] == 0].copy()
    if good.empty:
        raise ValueError("Keine guten Frames in der CSV.")
    for direction, part in good.groupby("direction"):
        fig, ax = plt.subplots(figsize=(11, 5))
        for (_, bottle), frames in part.groupby(["group_id", "bottle_id"]):
            frames = frames.sort_values("phase")
            ax.plot(frames.phase, frames.score, label=bottle, linewidth=1)
        ax.set(xlabel="Fahrphase", ylabel="Rohscore", title=str(direction))
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(out / f"rohscore_{direction}.png", dpi=150)
        plt.close(fig)
    peaks = good.loc[good.groupby(["group_id", "direction"]).score.idxmax()]
    peaks.to_csv(out / "rohscore_maxima.csv", index=False)


def compare_images(inputs, out, count, top_fraction):
    index = pd.read_csv(inputs / "index.csv")
    if not index.label.isin([0, 1]).all() or index.groupby("group_id").label.nunique().max() != 1:
        raise ValueError("Eindeutige Labels 0/1 je Gruppe erforderlich.")
    labels = index.groupby("group_id").label.first()
    good_ids = labels.index[labels == 0].tolist()
    bad_ids = labels.index[labels == 1].tolist()
    ids = good_ids + bad_ids
    if len(good_ids) < 2:
        raise ValueError("Mindestens zwei gute Flakons erforderlich.")
    if index.duplicated(["group_id", "direction"]).any():
        raise ValueError("Mehrere Aufnahmen pro Gruppe und Richtung. Messwiederholungen separat kennzeichnen.")
    phases = (np.arange(count) + 0.5) / count
    pairs = list(combinations(good_ids, 2)) + [(a, b) for a in bad_ids for b in good_ids]
    rows = []
    for direction in sorted(index.direction.unique()):
        frames = {}
        for r in index.loc[index.direction == direction].itertuples():
            arr = np.load(inputs / str(r.path).replace(chr(92), "/"), mmap_mode="r")
            if arr.ndim != 3 or len(arr) != r.n_frames or arr.dtype != np.uint8:
                raise ValueError(f"Erwartet uint8 (T,H,W) passend zu index.csv: {r.path}")
            # Nächster vorhandener Frame zu phase=t/n_frames (wie BottleDataset).
            t = np.minimum(np.floor(phases * len(arr) + 0.5).astype(int), len(arr) - 1)
            actual = t + int(r.lag)
            valid = (t >= r.valid_start) & (t < r.valid_end) & (actual >= 0) & (actual < len(arr))
            frames[r.group_id] = (arr, actual, valid)
        if set(frames) != set(ids):
            raise ValueError(f"Fehlende Aufnahme für Richtung {direction}.")
        for a, b in pairs:
            x, ix, vx = frames[a]
            y, iy, vy = frames[b]
            if x.shape[1:] != y.shape[1:]:
                raise ValueError(f"Bildgrößen unterscheiden sich: {a}, {b}")
            seen = set()
            for j in np.flatnonzero(vx & vy):
                key = (ix[j], iy[j])
                if key in seen:
                    continue
                seen.add(key)
                delta = np.abs(x[ix[j]].astype(np.int16) - y[iy[j]])
                rows.append((a, b, direction, phases[j], *key, delta.mean()))
            if not seen:
                raise ValueError(f"Keine gemeinsamen gültigen Phasen: {a}, {b}, {direction}")
    detail = pd.DataFrame(rows, columns=["a", "b", "direction", "phase", "frame_a", "frame_b", "mae"])
    detail.to_csv(out / "paarvergleich_phasen.csv", index=False)
    summary = []
    for (a, b, direction), part in detail.groupby(["a", "b", "direction"]):
        n_top = int(np.ceil(top_fraction * len(part)))
        summary.append((a, b, direction, len(part), part.mae.median(), part.mae.nlargest(n_top).mean()))
    sweeps = pd.DataFrame(summary, columns=["a", "b", "direction", "n", "typisch", "spitze"])
    sweeps.to_csv(out / "paarvergleich_richtungen.csv", index=False)
    avg = sweeps.groupby(["a", "b"])[["typisch", "spitze"]].mean()
    ranking = pd.DataFrame(index=pd.Index(ids, name="group_id"))
    ranking["label"] = labels
    ranking["n_referenzen"] = [len(good_ids) - int(labels[g] == 0) for g in ids]
    for metric in avg:
        matrix = pd.DataFrame(np.nan, index=ids, columns=good_ids)
        for (a, b), value in avg[metric].items():
            matrix.loc[a, b] = value
            if a in good_ids:
                matrix.loc[b, a] = value
        ranking[metric] = matrix.median(axis=1)
        ranking[f"{metric}_naechste3"] = matrix.apply(lambda row: row.nsmallest(3).mean(), axis=1)
        matrix.to_csv(out / f"matrix_{metric}.csv", index_label="group_id")
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(matrix, cmap="magma", vmin=0)
        if len(ids) <= 40:
            ax.set_xticks(range(len(good_ids)), good_ids, rotation=90, fontsize=6)
            ax.set_yticks(range(len(ids)), ids, fontsize=6)
        ax.set_title(f"Paarweise Bildabweichung: {metric}")
        fig.colorbar(im, ax=ax, label="Mittlere absolute Grauwertdifferenz (0–255)")
        fig.tight_layout()
        fig.savefig(out / f"matrix_{metric}.png", dpi=160)
        plt.close(fig)
    ranking.sort_values("typisch", ascending=False).to_csv(out / "flakonvergleich.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-scores", type=Path, help="frame_scores.csv aus einem einzelnen Modelllauf")
    parser.add_argument("--inputs", type=Path, help="Preprocessing-Ordner mit index.csv und NPY-Sequenzen")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phasen", type=int, default=50)
    parser.add_argument("--spitzenanteil", type=float, default=0.10)
    args = parser.parse_args()
    if not (args.inputs or args.frame_scores) or args.phasen < 1 or not 0 < args.spitzenanteil <= 1:
        parser.error("Mindestens eine Datenquelle, positive Phasenanzahl und Spitzenanteil in (0,1] erforderlich.")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.frame_scores:
        score_profiles(args.frame_scores, args.out)
    if args.inputs:
        compare_images(args.inputs, args.out, args.phasen, args.spitzenanteil)
