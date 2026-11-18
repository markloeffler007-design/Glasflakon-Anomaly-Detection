
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
