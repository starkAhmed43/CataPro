import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_HPARAMS_JSON = BENCH_DIR / "default_hparams.json"


def discover_split_jobs(value_root: Path, split_groups, explicit_thresholds=None):
    """Return list of (split_group, split_name, split_dir) for all discoverable jobs.

    Handles two layouts:
      - Direct: train/val/test files sit directly inside the split_group dir.
      - Thresholded: split_group contains threshold_* (or easy/medium/hard) subdirs.
    """
    jobs = []
    for split_group in split_groups:
        split_root = value_root / split_group
        if not split_root.exists():
            continue

        # Direct layout — files live right in the split group dir
        if _find_split_file(split_root, "train") and not explicit_thresholds:
            jobs.append((split_group, split_group, split_root))
            continue

        # Thresholded layout
        if explicit_thresholds:
            candidate_dirs = [split_root / t for t in explicit_thresholds]
        else:
            candidate_dirs = sorted(
                p for p in split_root.iterdir()
                if p.is_dir() and (p.name.startswith("threshold_") or p.name in {"easy", "medium", "hard"})
            )
        for d in candidate_dirs:
            if d.exists():
                jobs.append((split_group, d.name, d))
    return jobs


def _find_split_file(directory: Path, stem: str):
    for suffix in (".parquet", ".csv"):
        p = directory / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def ensure_split_triplet(threshold_dir: Path):
    return (
        _find_split_file(threshold_dir, "train"),
        _find_split_file(threshold_dir, "val"),
        _find_split_file(threshold_dir, "test"),
    )


def _threshold_to_float(name: str):
    try:
        return float(str(name).split("threshold_")[-1])
    except Exception:
        return float("inf")


def _slug(text: str):
    return str(text).replace("/", "_").replace(" ", "_")


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def get_split_meta(train_csv: Path, val_csv: Path, test_csv: Path, ratio_tolerance: float):
    train_size = len(_read_table(train_csv))
    val_size = len(_read_table(val_csv))
    test_size = len(_read_table(test_csv))
    total = train_size + val_size + test_size

    if total == 0:
        train_ratio = 0.0
        val_ratio = 0.0
        test_ratio = 0.0
    else:
        train_ratio = train_size / total
        val_ratio = val_size / total
        test_ratio = test_size / total

    target = (0.8, 0.1, 0.1)
    small_split_flag = int(
        abs(train_ratio - target[0]) > ratio_tolerance
        or abs(val_ratio - target[1]) > ratio_tolerance
        or abs(test_ratio - target[2]) > ratio_tolerance
    )

    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "small_split_flag": small_split_flag,
    }


def maybe_build_feature(csv_path, out_pkl, args):
    if out_pkl.exists() and not args.overwrite:
        return

    cmd = [
        sys.executable,
        "emulator_bench/build_tvt_features.py",
        "--input_csv",
        str(csv_path),
        "--output_pkl",
        str(out_pkl),
        "--target_col",
        args.target_col,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--prot_batch_size",
        str(args.prot_batch_size),
        "--mol_batch_size",
        str(args.mol_batch_size),
        "--cache_dir",
        args.cache_dir,
    ]

    if args.no_cache_read:
        cmd.append("--no_cache_read")
    if args.no_cache_write:
        cmd.append("--no_cache_write")

    subprocess.run(cmd, check=True)


def run_training(train_pkl, val_pkl, test_pkl, out_dir, args, task_name, seed):
    cmd = [
        sys.executable,
        "emulator_bench/train_single_target_tvt.py",
        "--train_pkl",
        str(train_pkl),
        "--val_pkl",
        str(val_pkl),
        "--test_pkl",
        str(test_pkl),
        "--out_dir",
        str(out_dir),
        "--task_name",
        task_name,
        "--batch_size",
        str(args.train_batch_size),
        "--lr",
        str(args.lr),
        "--drop_rate",
        str(args.drop_rate),
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--patience",
        str(args.patience),
        "--min_delta",
        str(args.min_delta),
        "--seed",
        str(seed),
    ]

    if args.skip_singleton_batch:
        cmd.append("--skip_singleton_batch")

    if args.no_early_stopping:
        cmd.append("--no_early_stopping")

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run CataPro TVT benchmark across split families and thresholds for one value type."
    )
    parser.add_argument(
        "--base_dir",
        default="/home/adhil/github/EMULaToR/data/processed/baselines/CataPro",
        type=str,
        help="Root directory containing kcat/km/ki folders.",
    )
    parser.add_argument("--value_type", default=None, choices=["kcat", "km", "ki"], type=str,
                        help="Value type subdirectory under base_dir. Omit if using --value_root.")
    parser.add_argument("--value_root", default=None, type=str,
                        help="Direct path to the split-group directory, bypassing base_dir/value_type.")
    parser.add_argument(
        "--split_groups",
        nargs="+",
        default=["enzyme_sequence_splits", "substrate_splits"],
        help="Split families under each value type folder.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=None,
        help="Optional explicit threshold directories (e.g. threshold_0.1 threshold_0.2).",
    )

    parser.add_argument("--target_col", default="log10_value", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)

    parser.add_argument("--prot_batch_size", default=32, type=int)
    parser.add_argument("--mol_batch_size", default=64, type=int)
    parser.add_argument("--cache_dir", default="emulator_bench/.cache_embeddings", type=str)
    parser.add_argument("--no_cache_read", action="store_true")
    parser.add_argument("--no_cache_write", action="store_true")

    parser.add_argument("--train_batch_size", default=2048, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--drop_rate", default=0.0, type=float)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--min_delta", default=0.001, type=float)
    parser.add_argument("--skip_singleton_batch", action="store_true")
    parser.add_argument("--no_early_stopping", "--no-early-stopping", action="store_true",
                        help="Disable early stopping and train for the full --epochs.")
    parser.add_argument(
        "--hparams_json",
        type=str,
        default=None,
        help="Optional JSON file with tuned hyperparameters to override train settings.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds for repeated runs per split.")
    parser.add_argument("--ratio_tolerance", type=float, default=0.02, help="Tolerance for 80:10:10 ratio flagging.")
    parser.add_argument(
        "--primary_metric",
        type=str,
        default="MSE",
        choices=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"],
        help="Metric used for ranking threshold summaries.",
    )
    parser.add_argument("--higher_is_better", action="store_true", help="Set ranking direction for primary metric.")

    parser.add_argument("--overwrite", action="store_true", help="Rebuild features/retrain even if outputs exist.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned jobs and exit.")

    args = parser.parse_args()

    hparams_path = Path(args.hparams_json) if args.hparams_json else DEFAULT_HPARAMS_JSON
    if hparams_path.exists():
        with open(hparams_path, "r") as f:
            payload = json.load(f)
        hp = payload.get("best_hparams", payload)

        key_map = {
            "train_batch_size": int,
            "lr": float,
            "drop_rate": float,
            "epochs": int,
            "patience": int,
            "min_delta": float,
        }
        for k, caster in key_map.items():
            if k in hp:
                setattr(args, k, caster(hp[k]))

        if "no_early_stopping" in hp:
            args.no_early_stopping = bool(hp["no_early_stopping"])

        if "split_groups" in hp and args.split_groups == parser.get_default("split_groups"):
            args.split_groups = list(hp["split_groups"])

        print(f"Loaded hyperparameters from {hparams_path}")

    if args.value_root:
        value_root = Path(args.value_root).expanduser()
        if args.value_type is None:
            args.value_type = value_root.name
    elif args.value_type:
        value_root = Path(args.base_dir).expanduser() / args.value_type
    else:
        raise ValueError("Provide --value_type or --value_root.")

    if not value_root.exists():
        raise FileNotFoundError(f"Value root directory not found: {value_root}")

    jobs = discover_split_jobs(value_root, args.split_groups, args.thresholds)
    if not jobs:
        raise RuntimeError("No threshold jobs discovered. Check --base_dir/--value_type/--split_groups/--thresholds")

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not-set>')}")
    print(f"Discovered {len(jobs)} jobs for value_type={args.value_type}")

    if args.dry_run:
        for split_group, threshold_name, threshold_dir in jobs:
            print(f"- {split_group}/{threshold_name}: {threshold_dir}")
        return

    run_rows = []
    progress = tqdm(jobs, desc=f"{args.value_type} benchmark", unit="job")

    for split_group, threshold_name, threshold_dir in progress:
        progress.set_postfix(split=split_group, threshold=threshold_name)

        train_csv, val_csv, test_csv = ensure_split_triplet(threshold_dir)
        if not (train_csv and val_csv and test_csv):
            print(f"[skip] missing train/val/test files in {threshold_dir}")
            continue

        split_meta = get_split_meta(train_csv, val_csv, test_csv, args.ratio_tolerance)

        feats_dir = threshold_dir / "catapro_features"
        out_dir = threshold_dir / "catapro_results"
        feats_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_pkl = feats_dir / "train_feats.pkl"
        val_pkl = feats_dir / "val_feats.pkl"
        test_pkl = feats_dir / "test_feats.pkl"

        maybe_build_feature(train_csv, train_pkl, args)
        maybe_build_feature(val_csv, val_pkl, args)
        maybe_build_feature(test_csv, test_pkl, args)

        for seed in args.seeds:
            seed_out_dir = out_dir / f"seed_{seed}"
            final_test_csv = seed_out_dir / "final_results_test.csv"

            if not final_test_csv.exists() or args.overwrite:
                task_name = f"{args.value_type}_{split_group}_{threshold_name}_seed{seed}"
                run_training(train_pkl, val_pkl, test_pkl, seed_out_dir, args, task_name, seed)

            if final_test_csv.exists():
                row = pd.read_csv(final_test_csv).iloc[0].to_dict()
                row["value_type"] = args.value_type
                row["split_group"] = split_group
                row["threshold"] = threshold_name
                row["seed"] = seed
                row["results_dir"] = str(seed_out_dir)
                row.update(split_meta)
                run_rows.append(row)

    if run_rows:
        runs_df = pd.DataFrame(run_rows)
        runs_df["threshold_num"] = runs_df["threshold"].map(_threshold_to_float)
        runs_df = runs_df.sort_values(["split_group", "threshold_num", "seed"]).drop(columns=["threshold_num"])

        runs_path = value_root / "catapro_summary_runs.csv"
        runs_df.to_csv(runs_path, index=False)

        for split_group, g_runs in runs_df.groupby("split_group", sort=False):
            split_slug = _slug(split_group)
            split_runs_path = value_root / f"catapro_summary_runs__{split_slug}.csv"
            g_runs.to_csv(split_runs_path, index=False)

        metric_cols = [c for c in ["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"] if c in runs_df.columns]
        group_cols = ["value_type", "split_group", "threshold"]

        threshold_rows = []
        for keys, g in runs_df.groupby(group_cols, sort=False):
            row = dict(zip(group_cols, keys))
            row["n_seeds"] = int(g["seed"].nunique())
            for c in ["train_size", "val_size", "test_size", "train_ratio", "val_ratio", "test_ratio", "small_split_flag"]:
                row[c] = g[c].iloc[0]
            for m in metric_cols:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_var"] = float(g[m].var(ddof=1)) if len(g) > 1 else 0.0
            threshold_rows.append(row)

        threshold_df = pd.DataFrame(threshold_rows)
        threshold_df["threshold_num"] = threshold_df["threshold"].map(_threshold_to_float)
        threshold_df = threshold_df.sort_values(["split_group", "threshold_num"]).drop(columns=["threshold_num"])

        threshold_path = value_root / "catapro_summary_thresholds.csv"
        threshold_df.to_csv(threshold_path, index=False)

        # Backward-compatible name points to threshold-level aggregate summary.
        compat_path = value_root / "catapro_summary.csv"
        threshold_df.to_csv(compat_path, index=False)

        for split_group, g_th in threshold_df.groupby("split_group", sort=False):
            split_slug = _slug(split_group)
            split_threshold_path = value_root / f"catapro_summary_thresholds__{split_slug}.csv"
            g_th.to_csv(split_threshold_path, index=False)

            split_compat_path = value_root / f"catapro_summary__{split_slug}.csv"
            g_th.to_csv(split_compat_path, index=False)

        by_split_rows = []
        for split_group, g in threshold_df.groupby("split_group", sort=False):
            row = {"value_type": args.value_type, "split_group": split_group, "n_thresholds": len(g)}
            for m in metric_cols:
                row[f"{m}_mean_over_thresholds"] = float(g[f"{m}_mean"].mean())
                row[f"{m}_var_over_thresholds"] = float(g[f"{m}_mean"].var(ddof=1)) if len(g) > 1 else 0.0
            by_split_rows.append(row)

        by_split_df = pd.DataFrame(by_split_rows)
        by_split_path = value_root / "catapro_summary_by_split_group.csv"
        by_split_df.to_csv(by_split_path, index=False)

        metric_key = f"{args.primary_metric}_mean"
        if metric_key in threshold_df.columns:
            ranked_df = threshold_df.sort_values(metric_key, ascending=not args.higher_is_better)
            ranked_path = value_root / "catapro_summary_ranked.csv"
            ranked_df.to_csv(ranked_path, index=False)

            for split_group, g_rank in ranked_df.groupby("split_group", sort=False):
                split_slug = _slug(split_group)
                split_ranked_path = value_root / f"catapro_summary_ranked__{split_slug}.csv"
                g_rank.to_csv(split_ranked_path, index=False)

        print(f"Saved runs summary: {runs_path}")
        print(f"Saved threshold summary: {threshold_path}")
        print(f"Saved split-group summary: {by_split_path}")
        print("Saved split-group-specific summaries: catapro_summary_*__<split_group>.csv")
    else:
        print("No completed jobs to summarize.")


if __name__ == "__main__":
    main()
