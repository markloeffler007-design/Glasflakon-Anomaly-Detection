"""Vergleich guter Flakons. Aufrufbeispiele über --help."""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
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


def save_frame_index_matrix(out, index, ids, good_ids, top_frames, bin_size):
    """Farben kodieren absolute Frame-Indexbereiche; die Position kodiert Rang und Richtung."""
    directions = list(index.direction.unique())
    directions.sort(key=lambda d: ({"forward": 0, "backward": 1}.get(d, 2), d))
    row_of = {g: i for i, g in enumerate(ids)}
    col_of = {g: i for i, g in enumerate(good_ids)}
    lane_of = {direction: i for i, direction in enumerate(directions)}
    ranks = 10
    lanes = len(directions)
    # Zwei benachbarte Farbfelder je Rang: Frame der Zeile, Frame der Spalte.
    cells = np.full((len(ids) * lanes, len(good_ids) * ranks * 2), np.nan)
    for r in top_frames.itertuples(index=False):
        entries = [(r.a, r.b, r.frame_a, r.frame_b)]
        if r.a in col_of:
            entries.append((r.b, r.a, r.frame_b, r.frame_a))
        for a, b, frame_a, frame_b in entries:
            y = row_of[a] * lanes + lane_of[r.direction]
            x = (col_of[b] * ranks + r.rang - 1) * 2
            cells[y, x:x + 2] = [frame_a // bin_size, frame_b // bin_size]

    last_frame = int(index.n_frames.max()) - 1
    bins = last_frame // bin_size + 1
    cmap = plt.get_cmap("turbo", bins).copy()
    cmap.set_bad("#f2f3f5")
    norm = BoundaryNorm(np.arange(bins + 1) - 0.5, cmap.N)
    fig, ax = plt.subplots(figsize=(max(9, len(good_ids) * 0.85 + 4),
                                    max(4.7, len(ids) * lanes * 0.22 + 3.7)),
                           constrained_layout=True)
    try:
        im = ax.imshow(cells, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
        ax.set_xticks((np.arange(len(good_ids)) + 0.5) * ranks * 2 - 0.5,
                      good_ids, rotation=90, fontsize=8)
        ax.set_yticks((np.arange(len(ids)) + 0.5) * lanes - 0.5, ids, fontsize=8)
        for edge in range(0, cells.shape[1], 2):
            pair_edge = edge % (ranks * 2) == 0
            ax.axvline(edge - 0.5, color="#686d74" if pair_edge else "white",
                       linewidth=0.8 if pair_edge else 0.35)
        for edge in range(0, cells.shape[0], lanes):
            ax.axhline(edge - 0.5, color="#686d74", linewidth=0.8)

        names = {"forward": "Hinfahrt", "backward": "Rückfahrt"}
        lane_labels = " / ".join(names.get(d, d) for d in directions)
        ax.set_title("Frame-Indexbereiche der zehn größten Grauwertabweichungen", pad=29)
        ax.text(0, 1.012, f"Je Paar von links: Rang 1 bis 10 (nach MAE)   ·   "
                f"Spuren von oben: {lane_labels}", transform=ax.transAxes, fontsize=9)
        # Auch lange Sequenzen erhalten eine lesbare, diskrete Farblegende.
        ticks = np.unique(np.linspace(0, bins - 1, min(12, bins), dtype=int))
        labels = []
        for group in ticks:
            start = group * bin_size
            end = min(start + bin_size - 1, last_frame)
            labels.append(str(start) if start == end else f"{start}–{end}")
        bar = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=ticks, fraction=0.07,
                           pad=0.05, aspect=50)
        bar.set_ticklabels(labels)
        bar.ax.tick_params(labelsize=8)
        bar.set_label("Frame-Indexbereich (ab 0)   ·   Je Rang linke Hälfte: Zeilenflakon, "
                      "rechte Hälfte: Spaltenflakon")
        fig.savefig(out / "matrix_top10_framebereiche.png", dpi=160)
    finally:
        plt.close(fig)


def compare_images(inputs, out, count, top_fraction, bin_size=20):
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

    # Eine Rangliste je Flakonpaar über alle Fahrtrichtungen.
    top_frames = (
        detail.sort_values(
            ["a", "b", "mae", "direction", "frame_a", "frame_b"],
            ascending=[True, True, False, True, True, True],
        )
        .groupby(["a", "b"], sort=False)
        .head(10)
        .copy()
    )
    top_frames.insert(2, "rang", top_frames.groupby(["a", "b"], sort=False).cumcount() + 1)
    top_frames.to_csv(out / "paarvergleich_top10_frames.csv", index=False)

    save_frame_index_matrix(out, index, ids, good_ids, top_frames, bin_size)

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
    parser.add_argument("--frame-gruppe", type=int, default=20, metavar="N",
                        help="Anzahl aufeinanderfolgender Frame-Indizes je Farbe (Standard: 20)")
    args = parser.parse_args()
    if args.frame_gruppe < 1:
        parser.error("--frame-gruppe muss mindestens 1 sein.")
    if not (args.inputs or args.frame_scores) or args.phasen < 1 or not 0 < args.spitzenanteil <= 1:
        parser.error("Mindestens eine Datenquelle, positive Phasenanzahl und Spitzenanteil in (0,1] erforderlich.")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.frame_scores:
        score_profiles(args.frame_scores, args.out)
    if args.inputs:
        compare_images(args.inputs, args.out, args.phasen, args.spitzenanteil, args.frame_gruppe)
