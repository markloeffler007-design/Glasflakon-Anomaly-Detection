#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose der AUROC-Inversion (EfficientAD, gilt genauso für die Referenzbaseline).

Arbeitet ausschließlich auf den bereits geschriebenen Frame-Score-CSVs; es wird
nichts trainiert und kein Modell geladen. Erwartete Spalten (wie in
referenzbaseline.py / efficientadmodell.py):
    bottle_id, group_id, direction, phase, label, score  [frame_index, score_z]

Eingaben
    --eval   frame_scores.csv des Test-/Eval-Laufs (mehrere erlaubt, z. B. CV-Folds)
    --val    val_frame_scores.csv (gute Validierungsgruppen aus dem Modellordner)
    --train  optional: Frame-Scores der Trainingsgruppen (guter Train-Fold), z. B. per
             "score --split trainval --folds <Trainingsfolds> --allow-leakage"
    --heatmaps optional: Ordner mit den Heatmaps aus "--save-maps"

Schritte (entsprechen der Diagnose-Reihenfolge)
    1  Score-Verteilungen: Train-Gut / Val-Gut / Eval-Gut / Eval-Fehler
    2  Phasenprofil: mittlerer Score je Phase, AUROC je Phase, Maxima-Verteilung
    3  Score gegen Zeitstempel (Drift) und Zusammenhang Label <-> Aufnahmezeit
    4  Heatmap-Auswahl: höchste Gut-Sweeps, niedrigste Fehler-Sweeps
    5  Datenchecks (NaN, Sättigung, gemischte Labels) und automatischer Befund

Beispiel
    python diagnose_inversion.py ^
        --eval D:/Results/final/frame_scores.csv ^
        --val  D:/Store/final/val_frame_scores.csv ^
        --out  D:/Diagnose/efficientad
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ============================================================
# KONFIGURATION
# ============================================================

POPS = ["Train-Gut", "Val-Gut", "Eval-Gut", "Eval-Fehler"]
COLORS = {
    "Train-Gut": "#7fb3d5",
    "Val-Gut": "#2e86c1",
    "Eval-Gut": "#1e8449",
    "Eval-Fehler": "#c0392b",
}
PAIRS = [
    ("Train-Gut", "Val-Gut"),
    ("Train-Gut", "Eval-Gut"),
    ("Val-Gut", "Eval-Gut"),
    ("Eval-Gut", "Eval-Fehler"),
    ("Val-Gut", "Eval-Fehler"),
]

# Heuristische Schwellen für den automatischen Befund (Orientierung, kein Test)
T_INVERS = 0.45          # mittlere Sweep-AUROC darunter: Inversion
T_KEIN_SIGNAL = 0.55     # bis hierhin: praktisch kein Signal
T_LUECKE = 0.65          # |AUROC - 0.5| >= 0.15 zwischen Gut-Populationen
T_ETA2 = 0.5             # Anteil der Varianz, den die Phase erklärt
T_PROFIL_RHO = 0.8       # Rangkorrelation der Phasenprofile Gut <-> Fehler
T_DIFF_ANTEIL = 0.7      # Anteil der Phasen-Bins mit Fehler < Gut
T_ARGMAX = 0.5           # Anteil der Sweep-Maxima in den 2 häufigsten Bins
T_ZEIT_AUROC = 0.30      # Label<->Zeit: AUROC außerhalb [0.30, 0.70]
T_ZEIT_RHO = 0.4         # |Spearman| Score <-> Zeit
T_SAETTIGUNG = 0.01      # Anteil der Frames exakt auf dem Maximalwert

TS_RE = re.compile(r"(\d{8})_(\d{6})")


# ============================================================
# EINLESEN
# ============================================================

def read_frames(path, score_col, prefix=None):
    df = pd.read_csv(path)
    missing = [c for c in ("bottle_id", "direction", "phase", "label", score_col)
               if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: Spalten fehlen: {missing}")
    if "group_id" not in df.columns:
        print(f"  Hinweis: {path} hat keine Spalte group_id; bottle_id dient als Gruppe.")
        df["group_id"] = df["bottle_id"]
    if "frame_index" not in df.columns:
        df["frame_index"] = df.groupby("bottle_id").cumcount()

    df["score"] = pd.to_numeric(df[score_col], errors="coerce")
    df["bottle_orig"] = df["bottle_id"].astype(str)
    df["bottle_id"] = df["bottle_orig"]
    df["group_id"] = df["group_id"].astype(str)
    df["direction"] = df["direction"].astype(str)
    df["label"] = df["label"].astype(int)
    df["phase"] = df["phase"].astype(float)
    df["frame_index"] = df["frame_index"].astype(int)
    if prefix:
        df["bottle_id"] = prefix + "/" + df["bottle_id"]
        df["group_id"] = prefix + "/" + df["group_id"]
    return df


def load_all(args):
    parts = []
    for i, path in enumerate(args.eval):
        d = read_frames(path, args.score_col, f"e{i}" if len(args.eval) > 1 else None)
        d["population"] = np.where(d["label"] == 1, "Eval-Fehler", "Eval-Gut")
        parts.append(d)
    for path, name, prefix in ((args.val, "Val-Gut", "val"), (args.train, "Train-Gut", "train")):
        if not path:
            continue
        d = read_frames(path, args.score_col, prefix)
        if (d["label"] == 1).any():
            print(f"  Hinweis: {path} enthält Fehler-Frames; sie werden verworfen.")
            d = d[d["label"] == 0].copy()
        d["population"] = name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ============================================================
# AGGREGATION
# ============================================================

def sweep_table(df, topk):
    rows = []
    for (pop, bid), g in df.groupby(["population", "bottle_id"], sort=False):
        s = g["score"].to_numpy(float)
        order = np.argsort(s)[::-1]
        rows.append({
            "population": pop,
            "bottle_id": bid,
            "bottle_orig": g["bottle_orig"].iloc[0],
            "group_id": g["group_id"].iloc[0],
            "direction": g["direction"].iloc[0],
            "label": int(g["label"].iloc[0]),
            "n_frames": int(len(g)),
            "score_max": float(s[order[0]]),
            "score_topk": float(s[order[:topk]].mean()),
            "argmax_phase": float(g["phase"].to_numpy()[order[0]]),
            "argmax_frame": int(g["frame_index"].to_numpy()[order[0]]),
        })
    return pd.DataFrame(rows)


def parse_ts(*texts):
    for t in texts:
        m = TS_RE.search(str(t))
        if m:
            try:
                return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")
            except ValueError:
                pass
    return pd.NaT


def group_table(sweeps):
    g = (
        sweeps.groupby(["population", "group_id"], sort=False)
        .agg(label=("label", "first"), n_sweeps=("bottle_id", "count"),
             score_max=("score_max", "mean"), score_topk=("score_topk", "mean"))
        .reset_index()
    )
    g["timestamp"] = [parse_ts(x) for x in g["group_id"]]
    return g.sort_values("score_topk", ascending=False).reset_index(drop=True)


def pair_auroc(a, b):
    """P(Score aus b > Score aus a)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    return float(roc_auc_score(y, np.r_[a, b]))


def describe(s):
    s = np.asarray(s, float)
    if len(s) == 0:
        return {}
    return {"n": len(s), "mean": s.mean(), "median": np.median(s),
            "p90": np.percentile(s, 90), "max": s.max()}


def summary_table(df, sweeps, groups):
    rows = []
    for pop in POPS:
        for level, vals in (
            ("frame", df.loc[df["population"] == pop, "score"]),
            ("sweep_max", sweeps.loc[sweeps["population"] == pop, "score_max"]),
            ("sweep_topk", sweeps.loc[sweeps["population"] == pop, "score_topk"]),
            ("gruppe_topk", groups.loc[groups["population"] == pop, "score_topk"]),
        ):
            d = describe(vals)
            if d:
                rows.append({"population": pop, "ebene": level, **d})
    return pd.DataFrame(rows)


def compare_table(sweeps, groups):
    rows = []
    for level, tab in (("sweep", sweeps), ("gruppe", groups)):
        for a, b in PAIRS:
            for col in ("score_max", "score_topk"):
                va = tab.loc[tab["population"] == a, col]
                vb = tab.loc[tab["population"] == b, col]
                if len(va) == 0 or len(vb) == 0:
                    continue
                rows.append({
                    "ebene": level, "a": a, "b": b, "score": col,
                    "n_a": len(va), "n_b": len(vb),
                    "auroc_P(b>a)": pair_auroc(va, vb),
                })
    return pd.DataFrame(rows)


def get_auroc(cmp, level, a, b, score):
    sel = cmp[(cmp["ebene"] == level) & (cmp["a"] == a) & (cmp["b"] == b) & (cmp["score"] == score)]
    return float(sel["auroc_P(b>a)"].iloc[0]) if len(sel) else np.nan


# ============================================================
# PHASE
# ============================================================

def add_bin(df, nbins):
    d = df.copy()
    d["bin"] = np.minimum((d["phase"].clip(0, 1) * nbins).astype(int), nbins - 1)
    return d


def phase_profile(df, nbins):
    d = add_bin(df, nbins)
    prof = d.groupby(["population", "direction", "bin"])["score"].agg(["mean", "std", "count"]).reset_index()
    rows = []
    ev = d[d["population"].isin(["Eval-Gut", "Eval-Fehler"])]
    for (dirn, b), g in ev.groupby(["direction", "bin"]):
        gg = g.loc[g["population"] == "Eval-Gut", "score"]
        ff = g.loc[g["population"] == "Eval-Fehler", "score"]
        rows.append({"direction": dirn, "bin": b, "auroc_frame": pair_auroc(gg, ff)})
    return prof, pd.DataFrame(rows)


def eta2(df, nbins, pops):
    d = add_bin(df[df["population"].isin(pops)], nbins)
    if len(d) < 10:
        return np.nan
    cell = d["direction"] + "|" + d["bin"].astype(str)
    y = d["score"].to_numpy(float)
    cm = d.groupby(cell)["score"].transform("mean").to_numpy(float)
    ss_t = ((y - y.mean()) ** 2).sum()
    return float(((cm - y.mean()) ** 2).sum() / ss_t) if ss_t > 0 else np.nan


def phase_stats(df, sweeps, prof, nbins):
    out = {}
    piv = prof.pivot_table(index=["direction", "bin"], columns="population", values="mean")

    def rho(a, b):
        if a in piv and b in piv:
            sub = piv[[a, b]].dropna()
            if len(sub) >= 5:
                return float(spearmanr(sub[a], sub[b]).correlation)
        return np.nan

    out["rho_gut_fehler"] = rho("Eval-Gut", "Eval-Fehler")
    out["rho_val_eval"] = rho("Val-Gut", "Eval-Gut")
    if "Eval-Gut" in piv and "Eval-Fehler" in piv:
        sub = piv[["Eval-Gut", "Eval-Fehler"]].dropna()
        out["anteil_fehler_niedriger"] = float((sub["Eval-Fehler"] < sub["Eval-Gut"]).mean()) if len(sub) else np.nan
    else:
        out["anteil_fehler_niedriger"] = np.nan
    out["eta2_gut"] = eta2(df, nbins, ["Train-Gut", "Val-Gut", "Eval-Gut"])
    out["eta2_fehler"] = eta2(df, nbins, ["Eval-Fehler"])

    ev = sweeps[sweeps["population"].isin(["Eval-Gut", "Eval-Fehler"])]
    if len(ev):
        b = np.minimum((ev["argmax_phase"].clip(0, 1) * nbins).astype(int), nbins - 1)
        top2 = b.value_counts().head(2).sum()
        out["argmax_top2_anteil"] = float(top2 / len(ev))
    else:
        out["argmax_top2_anteil"] = np.nan
    return out


# ============================================================
# ZEIT
# ============================================================

def time_stats(groups):
    g = groups.dropna(subset=["timestamp"]).copy()
    if len(g) < 4:
        return None
    g["t_h"] = (g["timestamp"] - g["timestamp"].min()).dt.total_seconds() / 3600.0
    out = {"bereiche": {}}
    for pop in POPS:
        s = g[g["population"] == pop]
        if len(s):
            out["bereiche"][pop] = (s["timestamp"].min(), s["timestamp"].max(),
                                    int(s["timestamp"].dt.date.nunique()), len(s))

    ev = g[g["population"].isin(["Eval-Gut", "Eval-Fehler"])]
    out["auroc_label_zeit"] = (
        float(roc_auc_score(ev["label"], ev["t_h"])) if ev["label"].nunique() == 2 else np.nan
    )

    def rho(sub):
        if len(sub) >= 4 and sub["t_h"].nunique() > 1:
            return float(spearmanr(sub["t_h"], sub["score_topk"]).correlation)
        return np.nan

    out["rho_gut"] = rho(g[g["population"].isin(["Train-Gut", "Val-Gut", "Eval-Gut"])])
    out["rho_eval_gut"] = rho(g[g["population"] == "Eval-Gut"])
    out["rho_fehler"] = rho(g[g["population"] == "Eval-Fehler"])
    return out


# ============================================================
# DATENCHECKS
# ============================================================

def sanity(df):
    s = df["score"]
    flags = set()
    lines = []

    n_bad = int((~np.isfinite(s)).sum())
    if n_bad:
        flags.add("nan")
        lines.append(f"[Hinweis] {n_bad} Frames mit NaN/inf-Score (werden verworfen).")
    fin = s[np.isfinite(s)]
    if len(fin):
        share_max = float((fin >= fin.max() - 1e-9).mean())
        share_zero = float((fin == 0).mean())
        lines.append(f"[Info] Score-Bereich {fin.min():.4g} … {fin.max():.4g}; "
                     f"{share_max:.2%} der Frames exakt auf dem Maximum, {share_zero:.2%} exakt 0.")
        if share_max > T_SAETTIGUNG:
            flags.add("saettigung")
            lines.append("[Hinweis] Viele Frames liegen exakt auf dem Maximalwert: "
                         "die Score-Normalisierung saturiert vermutlich (Clipping).")
        if (fin < 0).any():
            lines.append(f"[Info] {(fin < 0).mean():.2%} der Scores sind negativ (z. B. kalibrierte Scores).")

    per_sweep = df.groupby("bottle_id").size()
    lines.append(f"[Info] Frames je Sweep: min {per_sweep.min()}, Median {int(per_sweep.median())}, "
                 f"max {per_sweep.max()}; Phase {df['phase'].min():.3f} … {df['phase'].max():.3f}.")
    if per_sweep.nunique() > 1:
        lines.append("[Hinweis] Sweeps haben unterschiedlich viele Frames "
                     "(stride oder fehlende Frames?). Max/Top-k sind dann nicht direkt vergleichbar.")

    mixed = int((df.groupby("group_id")["label"].nunique() > 1).sum())
    if mixed:
        flags.add("gemischt")
        lines.append(f"[Hinweis] {mixed} Gruppe(n) mit gemischten Labels: Labelzuordnung prüfen.")
    return flags, lines


# ============================================================
# HEATMAP-AUSWAHL
# ============================================================

def safe_name(bottle_id):
    return str(bottle_id).replace("/", "_").replace("\\", "_")


def heatmap_targets(sweeps, n):
    gut = sweeps[sweeps["population"] == "Eval-Gut"].sort_values("score_topk", ascending=False).head(n).copy()
    feh = sweeps[sweeps["population"] == "Eval-Fehler"].sort_values("score_topk", ascending=True).head(n).copy()
    gut["art"] = "gut_hoch"
    feh["art"] = "fehler_niedrig"
    t = pd.concat([gut, feh], ignore_index=True)
    if t.empty:
        return t
    t["rang"] = t.groupby("art").cumcount() + 1
    t["heatmap_datei"] = [f"{safe_name(b)}_{int(f):06d}.png" for b, f in zip(t["bottle_orig"], t["argmax_frame"])]
    return t[["art", "rang", "bottle_orig", "group_id", "direction", "argmax_frame",
              "argmax_phase", "score_max", "score_topk", "heatmap_datei"]]


def copy_heatmaps(targets, src_dir, out_dir):
    if targets.empty:
        return 0, 0
    dst = Path(out_dir) / "heatmaps_selected"
    dst.mkdir(parents=True, exist_ok=True)
    found = missing = 0
    for _, r in targets.iterrows():
        src = Path(src_dir) / r["heatmap_datei"]
        if src.is_file():
            shutil.copy(src, dst / f"{r['art']}_{int(r['rang']):02d}_{r['heatmap_datei']}")
            found += 1
        else:
            missing += 1
    return found, missing


# ============================================================
# PLOTS
# ============================================================

def plot_distribution(df, sweeps, topk, path):
    pops = [p for p in POPS if (sweeps["population"] == p).any()]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for a, col, title in (
        (ax[0], "score_max", "Sweep: Maximum der Frame-Scores"),
        (ax[1], "score_topk", f"Sweep: Mittel der Top-{topk} Frame-Scores"),
    ):
        data = [sweeps.loc[sweeps["population"] == p, col].to_numpy() for p in pops]
        a.boxplot(data, positions=range(len(pops)), widths=0.5, showfliers=False)
        for i, (p, d) in enumerate(zip(pops, data)):
            a.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=24, color=COLORS[p], alpha=0.85, zorder=3)
        a.set_xticks(range(len(pops)))
        a.set_xticklabels([f"{p}\n(n={len(d)})" for p, d in zip(pops, data)], fontsize=9)
        a.set_title(title)
        a.set_ylabel("Score")
        a.grid(alpha=0.25)

    allv = df["score"].to_numpy(float)
    bins = np.linspace(np.nanmin(allv), np.nanpercentile(allv, 99.9), 60)
    for p in pops:
        ax[2].hist(df.loc[df["population"] == p, "score"], bins=bins, histtype="step",
                   density=True, color=COLORS[p], lw=1.8, label=p)
    ax[2].set_yscale("log")
    ax[2].set_title("Frame-Scores (Dichte, log)")
    ax[2].set_xlabel("Score")
    ax[2].legend(fontsize=9)
    ax[2].grid(alpha=0.25)
    fig.suptitle("1  Score-Verteilungen je Population", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_phase(prof, aur, nbins, path):
    dirs = sorted(prof["direction"].unique())
    fig, axes = plt.subplots(2, len(dirs), figsize=(7.5 * len(dirs), 7.5), squeeze=False, sharex=True)
    for j, d in enumerate(dirs):
        ax = axes[0, j]
        for p in POPS:
            s = prof[(prof["direction"] == d) & (prof["population"] == p)].sort_values("bin")
            if s.empty:
                continue
            x = (s["bin"].to_numpy() + 0.5) / nbins
            m = s["mean"].to_numpy(float)
            ax.plot(x, m, color=COLORS[p], label=p, lw=2.2 if p.startswith("Eval") else 1.2)
            if p.startswith("Eval"):
                sd = s["std"].fillna(0).to_numpy(float)
                ax.fill_between(x, m - sd, m + sd, color=COLORS[p], alpha=0.12)
        ax.set_title(f"Richtung: {d}")
        ax.set_ylabel("mittlerer Frame-Score")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=9)

        ax2 = axes[1, j]
        a = aur[aur["direction"] == d].sort_values("bin")
        if len(a):
            x = (a["bin"].to_numpy() + 0.5) / nbins
            v = a["auroc_frame"].to_numpy(float)
            ax2.bar(x, v - 0.5, bottom=0.5, width=0.8 / nbins,
                    color=np.where(v >= 0.5, "#1e8449", "#c0392b"))
        ax2.axhline(0.5, color="k", lw=1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Phase")
        ax2.set_ylabel("AUROC Fehler > Gut (Frames)")
        ax2.grid(alpha=0.25)
    fig.suptitle("2  Phasenprofil (oben) und AUROC je Phase (unten, Eval)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_argmax(sweeps, nbins, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, nbins + 1)
    for p in POPS:
        s = sweeps.loc[sweeps["population"] == p, "argmax_phase"]
        if len(s):
            ax.hist(s, bins=bins, histtype="step", density=True, color=COLORS[p], lw=1.8, label=f"{p} (n={len(s)})")
    ax.set_xlabel("Phase des höchsten Frame-Scores im Sweep")
    ax.set_ylabel("Dichte")
    ax.set_title("2b  Wo liegen die Sweep-Maxima?")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_time(groups, path):
    g = groups.dropna(subset=["timestamp"])
    if len(g) < 4:
        return False
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for p in POPS:
        s = g[g["population"] == p]
        if len(s):
            ax.scatter(s["timestamp"], s["score_topk"], s=45, color=COLORS[p], label=f"{p} (n={len(s)})",
                       marker="x" if p == "Eval-Fehler" else "o", alpha=0.9)
    ax.set_ylabel("Gruppen-Score (Top-k, Mittel der Sweeps)")
    ax.set_title("3  Score gegen Aufnahmezeit je Messung")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


# ============================================================
# BEFUND
# ============================================================

def fmt(v, nd=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def befund(cmp, ph, ts, san_flags, san_lines, populations):
    L = ["BEFUND (Heuristik, keine Statistik; bei wenigen Gruppen nur Hinweise)", "=" * 70]
    flags = set(san_flags)

    # Orientierung
    vals = [get_auroc(cmp, "sweep", "Eval-Gut", "Eval-Fehler", c) for c in ("score_max", "score_topk")]
    vals = [v for v in vals if not np.isnan(v)]
    if vals:
        m = float(np.mean(vals))
        if m < T_INVERS:
            flags.add("invers")
            L.append(f"[Hinweis] Sweep-AUROC Gut→Fehler = {fmt(m)}: Fehler-Sweeps werden im Mittel "
                     f"NIEDRIGER bewertet als gute (Inversion).")
        elif m <= T_KEIN_SIGNAL:
            L.append(f"[Hinweis] Sweep-AUROC Gut→Fehler = {fmt(m)}: praktisch kein Signal.")
        else:
            L.append(f"[ok] Sweep-AUROC Gut→Fehler = {fmt(m)}: Signal in der erwarteten Richtung.")
    else:
        L.append("[Info] Keine Eval-Fehlergruppen: Orientierung nicht prüfbar.")

    # Lücke zwischen guten Populationen
    for a, b in (("Train-Gut", "Val-Gut"), ("Val-Gut", "Eval-Gut"), ("Train-Gut", "Eval-Gut")):
        v = [get_auroc(cmp, "sweep", a, b, c) for c in ("score_max", "score_topk")]
        v = [x for x in v if not np.isnan(x)]
        if not v:
            continue
        m = float(np.mean(v))
        if abs(m - 0.5) >= T_LUECKE - 0.5:
            flags.add("luecke")
            richtung = f"{b} höher als {a}" if m > 0.5 else f"{a} höher als {b}"
            L.append(f"[Hinweis] {a} vs. {b}: AUROC {fmt(m)} ({richtung}). "
                     f"Auch gute Flaschen werden unterschiedlich bewertet: Generalisierungslücke.")
        else:
            L.append(f"[ok] {a} vs. {b}: AUROC {fmt(m)} (keine auffällige Lücke).")

    # Phase
    e = ph["eta2_gut"]
    if not np.isnan(e):
        if e >= T_ETA2:
            flags.add("phase")
            L.append(f"[Hinweis] Die Phase erklärt {e:.0%} der Varianz der Gut-Frame-Scores (η²): "
                     f"Phasen-Confound wahrscheinlich.")
        else:
            L.append(f"[ok] Die Phase erklärt {e:.0%} der Varianz der Gut-Frame-Scores (η²).")
    if not np.isnan(ph["rho_gut_fehler"]) and ph["rho_gut_fehler"] >= T_PROFIL_RHO:
        flags.add("phase")
        L.append(f"[Hinweis] Phasenprofile von Gut und Fehler fast gleich (Spearman {fmt(ph['rho_gut_fehler'])}): "
                 f"der Score folgt der Phase, nicht dem Fehler.")
    if not np.isnan(ph["anteil_fehler_niedriger"]) and ph["anteil_fehler_niedriger"] >= T_DIFF_ANTEIL:
        L.append(f"[Hinweis] In {ph['anteil_fehler_niedriger']:.0%} der Phasen-Bins liegen Fehler unter Gut: "
                 f"die Inversion ist breit, nicht lokal. Das spricht eher für Daten-/Normalisierungsursachen.")
    if not np.isnan(ph["argmax_top2_anteil"]) and ph["argmax_top2_anteil"] >= T_ARGMAX:
        flags.add("phase")
        L.append(f"[Hinweis] {ph['argmax_top2_anteil']:.0%} der Sweep-Maxima liegen in nur 2 Phasen-Bins: "
                 f"Max/Top-k messen vor allem 'schwierige Phasen'.")

    # Zeit
    if ts is None:
        L.append("[Info] Keine Zeitstempel in group_id (Muster JJJJMMTT_HHMMSS) gefunden: Drift-Check übersprungen.")
    else:
        a = ts["auroc_label_zeit"]
        if not np.isnan(a) and (a < T_ZEIT_AUROC or a > 1 - T_ZEIT_AUROC):
            flags.add("zeit")
            L.append(f"[Hinweis] Aufnahmezeit sagt das Label allein vorher (AUROC {fmt(a)}): "
                     f"Fehler und Gut wurden zu verschiedenen Zeiten gemessen; Drift und Label sind vermischt.")
        elif not np.isnan(a):
            L.append(f"[ok] Aufnahmezeit sagt das Label nicht vorher (AUROC {fmt(a)}).")
        for name, key in (("Gut", "rho_gut"), ("Eval-Gut", "rho_eval_gut"), ("Fehler", "rho_fehler")):
            r = ts[key]
            if not np.isnan(r) and abs(r) >= T_ZEIT_RHO:
                flags.add("zeit")
                L.append(f"[Hinweis] Score ↔ Zeit bei {name}: Spearman {fmt(r)}: Score driftet über die Messreihe.")

    L += [""] + san_lines

    L += ["", "EINORDNUNG", "=" * 70]
    mapping = [
        ("nan", "NaN/inf in den Scores: erst die Pipeline prüfen."),
        ("saettigung", "Saturierte Scores: Normalisierung/Clipping in der Modellausgabe prüfen."),
        ("gemischt", "Gemischte Labels in Gruppen: Labelzuordnung prüfen."),
        ("luecke", "Generalisierungslücke zwischen guten Flaschen: mehr und vielfältigere Gut-Daten, "
                   "Messvarianz untersuchen; Modellanpassung allein hilft nicht."),
        ("phase", "Phasen-Confound: Phase als Bedingung oder Kalibrierung ins Modell bringen "
                  "(z. B. phasenabhängige Normalisierung, score_z)."),
        ("zeit", "Drift/zeitlicher Confound: Messreihenfolge, Kalibrierung und Zuordnung Fehler/Gut über die Zeit prüfen; "
                 "ggf. Fehler- und Gut-Messungen zeitlich mischen."),
    ]
    hit = False
    for key, text in mapping:
        if key in flags:
            L.append(f"→ {text}")
            hit = True
    if "invers" in flags and not hit:
        L.append("→ Inversion ohne erkennbare Ursache in Verteilung, Phase und Zeit: "
                 "Vorzeichen/Normalisierung der Anomaliekarte im Code prüfen und die Heatmaps ansehen.")
    elif not hit:
        L.append("→ Keine der geprüften Ursachen fällt auf: Heatmaps in heatmaps_selected/ ansehen.")
    L.append("→ Vorzeichen und Normalisierung im Modellcode lassen sich aus CSVs nicht beurteilen "
             "(siehe Checkliste in der Ausgabe von --help).")
    return L


# ============================================================
# MAIN
# ============================================================

def create_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Checkliste Modellcode (nicht aus CSVs prüfbar):\n"
            "  - Wird die Anomaliekarte mit richtigem Vorzeichen bewertet (hoch = anomal)?\n"
            "  - Auf welchen Daten werden Normalisierungsquantile (q_a, q_b) berechnet,\n"
            "    und werden Scores danach geclippt?\n"
            "  - Stimmen Train-/Eval-Vorverarbeitung überein (Normierung, Größe, Graustufen)?\n"
            "  - Sind Hin- und Rückfahrt gleich behandelt (Phase, Richtung)?\n"
        ),
    )
    p.add_argument("--eval", nargs="+", required=True, help="frame_scores.csv des Eval-Laufs (mehrere erlaubt)")
    p.add_argument("--val", default=None, help="val_frame_scores.csv (gute Validierungsgruppen)")
    p.add_argument("--train", default=None, help="Frame-Scores guter Trainingsgruppen (optional)")
    p.add_argument("--heatmaps", default=None, help="Ordner mit Heatmaps (aus --save-maps)")
    p.add_argument("--out", required=True, help="Ausgabeordner")
    p.add_argument("--score-col", default="score", help="Score-Spalte (z. B. score oder score_z)")
    p.add_argument("--topk", type=int, default=5, help="Frames je Sweep für den Top-k-Score")
    p.add_argument("--phase-bins", type=int, default=20)
    p.add_argument("--n-heatmaps", type=int, default=5, help="Anzahl Sweeps je Heatmap-Gruppe")
    return p


def main():
    args = create_parser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_all(args)
    san_flags, san_lines = sanity(df)
    df = df[np.isfinite(df["score"])].copy()

    sweeps = sweep_table(df, args.topk)
    groups = group_table(sweeps)

    summ = summary_table(df, sweeps, groups)
    cmp = compare_table(sweeps, groups)
    prof, aur = phase_profile(df, args.phase_bins)
    ph = phase_stats(df, sweeps, prof, args.phase_bins)
    ts = time_stats(groups)
    targets = heatmap_targets(sweeps, args.n_heatmaps)

    summ.to_csv(out / "01_populationen_zusammenfassung.csv", index=False)
    cmp.to_csv(out / "01_populationen_auroc.csv", index=False)
    prof.to_csv(out / "02_phasenprofil.csv", index=False)
    aur.to_csv(out / "02_phase_auroc.csv", index=False)
    sweeps.to_csv(out / "sweeps.csv", index=False)
    groups.to_csv(out / "03_gruppen.csv", index=False)
    targets.to_csv(out / "04_heatmap_auswahl.csv", index=False)

    plot_distribution(df, sweeps, args.topk, out / "01_verteilung.png")
    plot_phase(prof, aur, args.phase_bins, out / "02_phasenprofil.png")
    plot_argmax(sweeps, args.phase_bins, out / "02b_maxima_phase.png")
    has_time = plot_time(groups, out / "03_zeit.png")

    if args.heatmaps:
        found, missing = copy_heatmaps(targets, args.heatmaps, out)
        print(f"Heatmaps: {found} kopiert, {missing} nicht gefunden "
              f"(erwartetes Namensmuster: <bottle_id mit / -> _>_<Frame 6-stellig>.png).")
    else:
        print("Heatmaps: nicht angegeben. 04_heatmap_auswahl.csv nennt die Frames; "
              "Auswertung mit --save-maps erzeugen und --heatmaps <Ordner> setzen.")

    lines = befund(cmp, ph, ts, san_flags, san_lines, sorted(df["population"].unique()))
    (out / "befund.txt").write_text("\n".join(lines), encoding="utf-8")

    pd.set_option("display.width", 160)
    print("\n" + "=" * 70)
    print("POPULATIONEN (Sweep-Ebene, AUROC P(b > a))")
    print("=" * 70)
    show = cmp[cmp["ebene"] == "sweep"]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + "\n".join(lines))
    print(f"\nErgebnisse: {out.resolve()}")
    if not has_time:
        print("(03_zeit.png nicht erzeugt: zu wenige Gruppen mit Zeitstempel)")


if __name__ == "__main__":
    main()
