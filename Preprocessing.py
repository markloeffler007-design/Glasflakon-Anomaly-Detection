#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocessing für Glasflakon-Aufnahmen.

Erwartete Ordnerstruktur:

ROOT/
    good/
        FLASCHE_001/
            cycle_01_forward/
                img_0001.png
                img_0002.png
            cycle_01_backward/
                img_0001.png
                img_0002.png

    defect/
        FLASCHE_002/
            cycle_01_forward/
            cycle_01_backward/

Jede Richtung wird als eigene Sequenz gespeichert.
Beide Richtungen einer Messung erhalten dieselbe group_id
und werden somit demselben Fold zugeordnet.

Ausgabe:
    Input/
        *.npy
        index.csv
        stats.json
        config.json
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. EINSTELLUNGEN
# ============================================================

# Hier deine absoluten Pfade eintragen:

DEFAULT_ROOT = Path(r"D:\1.Glass")
DEFAULT_OUT = Path(r"C:\Glasflakons\Input")

# Standard-Bildgröße
DEFAULT_OUT_SIZE = (270, 480)

# Standard-Farbkanal: "red" oder "gray"
DEFAULT_CHANNEL = "red"

# Anteil der Daten für den Testdatensatz
DEFAULT_TEST_FRACTION = 0.2

# Anzahl der Folds
DEFAULT_FOLDS = 5

# Zufallsstartwert
DEFAULT_SEED = 42


# ============================================================
# 2. ALLGEMEINE EINSTELLUNGEN
# ============================================================

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}

DIRECTION_RE = re.compile(
    r"cycle_\d+_(forward|backward)$",
    re.IGNORECASE
)


# ============================================================
# 3. BILDNUMMER ERMITTELN
# ============================================================

def frame_number(path):
    """
    Ermittelt die Bildnummer aus dem Dateinamen.

    Beispiel:
        img_0001.png -> 1
        img_0102.png -> 102
    """

    numbers = re.findall(r"\d+", path.stem)

    return int(numbers[-1]) if numbers else -1


# ============================================================
# 4. LABEL ERMITTELN
# ============================================================

def normalize_label(value):
    """
    Wandelt verschiedene Label-Bezeichnungen in Zahlen um.

    0 = gut
    1 = defekt
    """

    if isinstance(value, str):

        value = value.strip().lower()

        if value in {
            "0", "good", "ok", "gut", "io", "normal"
        }:
            return 0

        if value in {
            "1", "defect", "defekt", "bad",
            "nok", "fehler", "anomaly"
        }:
            return 1

    if pd.notna(value) and int(value) in (0, 1):
        return int(value)

    raise ValueError(
        f"Ungültiges Label: {value}. "
        "Erlaubt sind 0/good und 1/defect."
    )


def resolve_label(session_dir, root, labels_csv=None):
    """
    Ermittelt das Label anhand von:

    1. Optionaler labels.csv
    2. good-/defect-Ordner im relativen Datenpfad
    """

    rel_session = session_dir.relative_to(root).as_posix()
    group_id = rel_session.replace("/", "__")

    # --------------------------------------------------------
    # Label aus labels.csv
    # --------------------------------------------------------

    if labels_csv is not None:

        if "label" not in labels_csv.columns:
            raise ValueError(
                "Die labels.csv benötigt eine Spalte 'label'."
            )

        for key in (
            "group_id",
            "bottle_id",
            "session",
            "path",
            "src_dir"
        ):

            if key not in labels_csv.columns:
                continue

            values = (
                labels_csv[key]
                .astype(str)
                .str.replace("\\", "/", regex=False)
            )

            candidates = {
                rel_session,
                group_id,
                session_dir.name,
                str(session_dir.relative_to(root))
            }

            matches = labels_csv[values.isin(candidates)]

            if not matches.empty:

                label_value = matches.iloc[0]["label"]

                if pd.notna(label_value):
                    return normalize_label(label_value)

    # --------------------------------------------------------
    # Label anhand der Ordnerstruktur
    # --------------------------------------------------------

    parts = [
        part.lower()
        for part in Path(rel_session).parts
    ]

    for part in parts:

        if part in {
            "good",
            "ok",
            "gut",
            "io"
        }:
            return 0

        if part in {
            "defect",
            "defekt",
            "bad",
            "nok",
            "fehler"
        }:
            return 1

        if part.startswith((
            "defect_",
            "defekt_",
            "fehler_"
        )):
            return 1

    raise ValueError(
        f"Label für '{session_dir}' nicht erkannt. "
        "Bitte good/defect im Datenpfad verwenden "
        "oder eine labels.csv angeben."
    )


# ============================================================
# 5. RICHTUNGSORDNER FINDEN
# ============================================================

def find_direction_dirs(root):
    """
    Sucht rekursiv nach Ordnern wie:

        cycle_01_forward
        cycle_01_backward
        cycle_02_forward
        cycle_02_backward
    """

    return sorted(
        p for p in root.rglob("*")
        if p.is_dir() and DIRECTION_RE.match(p.name)
    )


# ============================================================
# 6. BILDSEQUENZ LADEN UND AUFBEREITEN
# ============================================================

def load_sequence(
    folder,
    roi=None,
    out_size=(270, 480),
    channel="red"
):
    """
    Liest sämtliche Bilder eines Richtungsordners ein.

    Verarbeitung:
        1. Bilder numerisch sortieren
        2. Bild einlesen
        3. Farbkanal auswählen
        4. Optional ROI ausschneiden
        5. Bildgröße vereinheitlichen
        6. Bilder zu einer Sequenz zusammenfassen

    Rückgabe:
        Array mit Form (T, H, W)
    """

    files = sorted(
        (
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_SUFFIXES
            and not p.name.startswith("._")
            and not p.name.startswith(".")
        ),
        key=lambda p: (
            frame_number(p),
            p.name.lower()
        )
    )

    if not files:
        raise ValueError(
            f"Keine Bilddateien in {folder}"
        )

    frames = []

    for path in files:

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                f"Bild kann nicht gelesen werden: {path}"
            )

        # ----------------------------------------------------
        # Farbkanal auswählen
        # ----------------------------------------------------

        if channel == "red":

            # OpenCV verwendet BGR:
            # Index 2 entspricht Rot.

            image = image[:, :, 2]

        elif channel == "gray":

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        else:
            raise ValueError(
                f"Unbekannter Farbkanal: {channel}"
            )

        # ----------------------------------------------------
        # ROI ausschneiden
        # ----------------------------------------------------

        if roi is not None:

            y0, y1, x0, x1 = roi

            if not (
                0 <= y0 < y1 <= image.shape[0]
                and 0 <= x0 < x1 <= image.shape[1]
            ):
                raise ValueError(
                    f"ROI {roi} liegt außerhalb des Bildes "
                    f"{path} mit Größe {image.shape}."
                )

            image = image[y0:y1, x0:x1]

        # ----------------------------------------------------
        # Bildgröße anpassen
        # ----------------------------------------------------

        width, height = out_size

        image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

        frames.append(image)

    return np.stack(frames).astype(np.uint8), files


# ============================================================
# 7. TRAINING-/TESTSPLIT UND FOLDS
# ============================================================

def assign_splits_defects_in_test(df, test_fraction, n_folds, seed):
    """
    Split-Modus "defects_in_test" (für unüberwachte Modelle mit wenigen Fehlern):

    - ALLE fehlerhaften Gruppen kommen ins Testset.
    - Von den guten Gruppen kommen test_fraction (mindestens 1) ins Testset.
    - Die übrigen guten Gruppen bilden trainval und werden auf die Folds verteilt.
    - trainval enthält damit nur gute Gruppen.
    Die Aufteilung erfolgt nach group_id (Hin- und Rückfahrt bleiben zusammen).
    """
    label_counts = df.groupby("group_id")["label"].nunique()
    if (label_counts > 1).any():
        raise ValueError("Mindestens eine group_id besitzt mehrere Labels.")

    groups = (
        df[["group_id", "label"]]
        .drop_duplicates("group_id")
        .reset_index(drop=True)
    )
    good_ids = groups.loc[groups["label"] == 0, "group_id"].to_numpy().copy()
    bad_ids = groups.loc[groups["label"] != 0, "group_id"].to_numpy().copy()

    n_good_test = int(round(test_fraction * len(good_ids))) if test_fraction > 0 else 0
    if test_fraction > 0:
        n_good_test = max(1, n_good_test)
    n_good_train = len(good_ids) - n_good_test
    if n_good_train < n_folds:
        raise ValueError(
            f"{len(good_ids)} gute Gruppen: nach {n_good_test} Test-Gruppen bleiben "
            f"{n_good_train}, aber {n_folds} Folds verlangt. "
            "Reduziere --folds oder --test-fraction."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(good_ids)
    test_good = set(good_ids[:n_good_test])
    train_good = good_ids[n_good_test:]

    df["split"] = "trainval"
    df["fold"] = -1
    df.loc[df["group_id"].isin(test_good | set(bad_ids)), "split"] = "test"

    for j, group_id in enumerate(train_good):
        df.loc[df["group_id"] == group_id, "fold"] = j % n_folds

    print(
        f"Split (defects_in_test): gute Gruppen Train/Val={len(train_good)}, "
        f"gute Gruppen Test={len(test_good)}, fehlerhafte Gruppen Test={len(bad_ids)}"
    )
    return df


def assign_splits(df, test_fraction, n_folds, seed, split_mode="stratified"):
    """
    Erstellt einen festen Testsplit.

    Die Aufteilung erfolgt anhand der group_id.

    Dadurch bleiben Vorwärts- und Rückwärtssequenzen
    derselben Messung immer zusammen.

    Die Folds werden innerhalb des trainval-Anteils
    annähernd klassenweise verteilt.
    """

    if split_mode == "defects_in_test":
        return assign_splits_defects_in_test(df, test_fraction, n_folds, seed)
    if split_mode != "stratified":
        raise ValueError(f"Unbekannter split_mode: {split_mode}")

    # --------------------------------------------------------
    # Prüfen, ob jede Gruppe nur ein Label besitzt
    # --------------------------------------------------------

    label_counts = df.groupby("group_id")["label"].nunique()

    if (label_counts > 1).any():
        raise ValueError(
            "Mindestens eine group_id besitzt mehrere Labels."
        )

    groups = (
        df[["group_id", "label"]]
        .drop_duplicates("group_id")
        .reset_index(drop=True)
    )

    if len(groups) < 2 and test_fraction > 0:
        raise ValueError(
            "Für einen Testsplit werden mindestens "
            "zwei unterschiedliche Gruppen benötigt."
        )

    # --------------------------------------------------------
    # Standardwerte
    # --------------------------------------------------------

    df["split"] = "trainval"
    df["fold"] = -1

    # --------------------------------------------------------
    # Testsplit erstellen
    # --------------------------------------------------------

    if test_fraction > 0:

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=seed
        )

        train_idx, test_idx = next(
            splitter.split(
                groups,
                groups["label"],
                groups["group_id"]
            )
        )

        train_groups = set(
            groups.iloc[train_idx]["group_id"]
        )

        test_groups = set(
            groups.iloc[test_idx]["group_id"]
        )

        df.loc[
            df["group_id"].isin(test_groups),
            "split"
        ] = "test"

    else:

        train_groups = set(groups["group_id"])

    # --------------------------------------------------------
    # Fold-Zuordnung
    # --------------------------------------------------------

    train_df = groups[
        groups["group_id"].isin(train_groups)
    ].copy()

    if len(train_df) < n_folds:

        raise ValueError(
            f"Nur {len(train_df)} Gruppen im trainval-Anteil, "
            f"aber {n_folds} Folds verlangt. "
            "Reduziere den Testanteil oder die Fold-Anzahl."
        )

    rng = np.random.default_rng(seed)

    fold_groups = {
        i: [] for i in range(n_folds)
    }

    for label in sorted(train_df["label"].unique()):

        ids = train_df.loc[
            train_df["label"] == label,
            "group_id"
        ].to_numpy().copy()

        rng.shuffle(ids)

        for j, group_id in enumerate(ids):

            fold = j % n_folds

            fold_groups[fold].append(group_id)

    for fold, ids in fold_groups.items():

        df.loc[
            df["group_id"].isin(ids),
            "fold"
        ] = fold

    return df


# ============================================================
# 8. HAUPTPROGRAMM
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --------------------------------------------------------
    # Pfade
    # --------------------------------------------------------

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Absoluter Pfad zum Rohdatenordner"
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Absoluter Pfad zum Ausgabeordner"
    )

    # --------------------------------------------------------
    # Optionale Einstellungen
    # --------------------------------------------------------

    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optionale CSV mit bottle_id und label"
    )

    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=None,
        metavar=("Y0", "Y1", "X0", "X1")
    )

    parser.add_argument(
        "--out-size",
        type=int,
        nargs=2,
        default=DEFAULT_OUT_SIZE,
        metavar=("WIDTH", "HEIGHT")
    )

    parser.add_argument(
        "--channel",
        choices=["red", "gray"],
        default=DEFAULT_CHANNEL
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION
    )

    parser.add_argument(
        "--split-mode",
        choices=["stratified", "defects_in_test"],
        default="stratified",
        help=(
            "stratified: Test- und Fold-Split über alle Gruppen (bisheriges Verhalten). "
            "defects_in_test: alle fehlerhaften Gruppen ins Testset, --test-fraction "
            "gilt nur für die guten Gruppen; trainval/Folds enthalten nur gute Gruppen."
        )
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Pfade überprüfen
    # --------------------------------------------------------

    root = args.root.resolve()
    out = args.out.resolve()

    print("\n====================================")
    print("GLASFLAKON PREPROCESSING")
    print("====================================")

    print(f"Rohdaten:  {root}")
    print(f"Ausgabe:   {out}")

    if not root.is_dir():

        raise SystemExit(
            f"\nRohdatenordner existiert nicht:\n{root}"
        )

    if root == out or root in out.parents:

        raise SystemExit(
            "Der Ausgabeordner darf nicht innerhalb "
            "des Rohdatenordners liegen. "
            "Bitte verwende einen separaten Ausgabeordner."
        )

    if not 0 <= args.test_fraction < 1:

        raise SystemExit(
            "--test-fraction muss zwischen 0 und 1 liegen."
        )

    if args.folds < 2:

        raise SystemExit(
            "--folds muss mindestens 2 sein."
        )

    if args.out_size[0] <= 0 or args.out_size[1] <= 0:

        raise SystemExit(
            "Die Bildabmessungen müssen größer als 0 sein."
        )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Optionale Labeldatei laden
    # --------------------------------------------------------

    labels_table = None

    if args.labels is not None:

        if not args.labels.is_file():

            raise SystemExit(
                f"Labels-Datei existiert nicht: {args.labels}"
            )

        labels_table = pd.read_csv(args.labels)

    # --------------------------------------------------------
    # Richtungsordner finden
    # --------------------------------------------------------

    direction_dirs = find_direction_dirs(root)

    if not direction_dirs:

        raise SystemExit(
            f"Keine cycle_N_forward/backward-Ordner "
            f"unter {root} gefunden."
        )

    print(
        f"\nGefundene Richtungsordner: {len(direction_dirs)}"
    )

    # --------------------------------------------------------
    # Sequenzen verarbeiten
    # --------------------------------------------------------

    rows = []

    for direction_dir in direction_dirs:

        session_dir = direction_dir.parent

        rel_session = (
            session_dir.relative_to(root).as_posix()
        )

        # Beide Richtungen einer Messung erhalten
        # dieselbe group_id.

        group_id = rel_session.replace("/", "__")

        direction_match = DIRECTION_RE.match(
            direction_dir.name
        )

        direction = direction_match.group(1).lower()

        label = resolve_label(
            session_dir,
            root,
            labels_table
        )

        # ----------------------------------------------------
        # Bilder laden und aufbereiten
        # ----------------------------------------------------

        arr, source_files = load_sequence(
            direction_dir,
            roi=args.roi,
            out_size=tuple(args.out_size),
            channel=args.channel
        )

        # ----------------------------------------------------
        # Dateinamen erstellen
        # ----------------------------------------------------

        safe_group = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            group_id
        )

        bottle_id = f"{safe_group}__{direction}"

        filename = f"{bottle_id}.npy"

        # ----------------------------------------------------
        # Numpy-Datei speichern
        # ----------------------------------------------------

        np.save(
            out / filename,
            arr
        )

        # ----------------------------------------------------
        # Indexeintrag erstellen
        # ----------------------------------------------------

        rows.append({

            "bottle_id": bottle_id,

            "group_id": group_id,

            "direction": direction,

            "path": filename,

            "label": label,

            "n_frames": int(len(arr)),

            "valid_start": 0,

            "valid_end": int(len(arr)),

            "lag": 0,

            "src_dir": str(
                direction_dir.relative_to(root)
            ),

            "session": rel_session

        })

        print(
            f"{bottle_id}: "
            f"{len(arr)} Frames | "
            f"Label={label}"
        )

    # --------------------------------------------------------
    # DataFrame erstellen
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Prüfen, ob Richtungen korrekt zugeordnet sind
    # --------------------------------------------------------

    if df.empty:

        raise SystemExit(
            "Es wurden keine Bildsequenzen verarbeitet."
        )

    # --------------------------------------------------------
    # Training-/Testsplit und Folds
    # --------------------------------------------------------

    df = assign_splits(
        df,
        args.test_fraction,
        args.folds,
        args.seed,
        split_mode=args.split_mode
    )

    # --------------------------------------------------------
    # Indexdatei speichern
    # --------------------------------------------------------

    df.to_csv(
        out / "index.csv",
        index=False
    )

    print("\nIndexdatei erstellt.")

    # --------------------------------------------------------
    # Statistiken für die Normalisierung
    # --------------------------------------------------------

    # Ausschließlich gute Sequenzen aus trainval verwenden.
    # Testdaten werden nicht für die Statistik verwendet.

    good_train = df[
        (df["split"] == "trainval")
        & (df["label"] == 0)
    ]

    if good_train.empty:

        raise SystemExit(
            "Keine guten Sequenzen im trainval-Anteil "
            "für die Normalisierungsstatistik vorhanden."
        )

    total_sum = 0.0
    total_sq = 0.0
    total_n = 0

    print("\nBerechne Normalisierungsstatistik ...")

    for row in good_train.itertuples():

        array = np.load(
            out / row.path,
            mmap_mode="r"
        )

        x = np.asarray(
            array,
            dtype=np.float64
        ) / 255.0

        total_sum += x.sum()

        total_sq += np.square(x).sum()

        total_n += x.size

    mean = total_sum / total_n

    variance = max(
        total_sq / total_n - mean * mean,
        1e-12
    )

    std = float(
        np.sqrt(variance)
    )

    stats = {

        "mean": float(mean),

        "std": std,

        "computed_from": "trainval good sequences",

        "pixel_count": int(total_n)

    }

    with open(
        out / "stats.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            stats,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Konfiguration speichern
    # --------------------------------------------------------

    config = {

        "root": str(root),

        "channel": args.channel,

        "roi": args.roi,

        "out_size_width_height": list(args.out_size),

        "test_fraction": args.test_fraction,

        "split_mode": args.split_mode,

        "folds": args.folds,

        "seed": args.seed,

        "lag": 0,

        "directions_separate": True,

        "grouped_by_session": True

    }

    with open(
        out / "config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Abschlussübersicht
    # --------------------------------------------------------

    print("\n====================================")
    print("PREPROCESSING ABGESCHLOSSEN")
    print("====================================")

    print(
        f"Sequenzen: {len(df)}"
    )

    print(
        f"Messungen/Flaschengruppen: "
        f"{df['group_id'].nunique()}"
    )

    print("\nVerteilung nach Split und Label:")

    print(
        df.groupby(
            ["split", "label"]
        ).size().to_string()
    )

    print("\nFold-Verteilung:")

    print(
        df.groupby(
            ["split", "fold"]
        ).size().to_string()
    )

    print(f"\nAusgabeordner: {out}")

    print("\nErstellte Dateien:")

    print("  - *.npy")
    print("  - index.csv")
    print("  - stats.json")
    print("  - config.json")

    print("\nFertig!")


# ============================================================
# 9. PROGRAMMSTART
# ============================================================

if __name__ == "__main__":
    main()
