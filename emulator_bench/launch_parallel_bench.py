"""
Parallel CataPro benchmark launcher.

Features are built sequentially (shared embedding cache is not concurrent-safe),
then all (split, seed) training jobs are dispatched across GPUs in parallel using
a thread-pool + work queue — N runs per GPU at a time.

Usage example:
    python emulator_bench/launch_parallel_bench.py \
        --value_type kcat \
        --gpus 0 1 2 3 \
        --runs_per_gpu 2 \
        --seeds 42 123 456
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
DEFAULT_HPARAMS_JSON = BENCH_DIR / "default_hparams.json"

BUILD_SCRIPT = BENCH_DIR / "build_tvt_features.py"
TRAIN_SCRIPT = BENCH_DIR / "train_single_target_tvt.py"
RANDOM_SPLIT_GROUP_ALIAS = "random_splits"
RANDOM_SPLIT_GROUP_PREFIX = "random_splits_grouped_"


# ---------------------------------------------------------------------------
# Helpers shared with run_split_benchmarks (inlined to avoid circular import)
# ---------------------------------------------------------------------------

def _threshold_to_float(name: str):
    try:
        return float(str(name).split("threshold_")[-1])
    except Exception:
        return float("inf")


def _slug(text: str):
    return str(text).replace("/", "_").replace(" ", "_")


def _find_split_file(directory: Path, stem: str):
    for suffix in (".parquet", ".csv"):
        p = directory / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def is_random_split_group(split_group: str):
    split_group = str(split_group)
    return split_group == RANDOM_SPLIT_GROUP_ALIAS or split_group.startswith(RANDOM_SPLIT_GROUP_PREFIX)


def expand_split_groups(value_root: Path, split_groups):
    expanded = []
    seen = set()
    grouped_random_dirs = None

    def add(split_group):
        if split_group not in seen:
            seen.add(split_group)
            expanded.append(split_group)

    for split_group in split_groups:
        split_group = str(split_group)
        if split_group != RANDOM_SPLIT_GROUP_ALIAS:
            add(split_group)
            continue

        if grouped_random_dirs is None:
            grouped_random_dirs = sorted(
                child.name
                for child in Path(value_root).glob(f"{RANDOM_SPLIT_GROUP_PREFIX}*")
                if child.is_dir()
            )

        if grouped_random_dirs:
            for grouped_split_group in grouped_random_dirs:
                add(grouped_split_group)
        elif (Path(value_root) / RANDOM_SPLIT_GROUP_ALIAS).exists():
            add(RANDOM_SPLIT_GROUP_ALIAS)

    return expanded


def discover_split_jobs(value_root: Path, split_groups, explicit_thresholds=None):
    """Return list of (split_group, split_name, split_dir) for all discoverable jobs.

    Handles two layouts:
      - Direct: train/val/test files sit directly inside the split_group dir.
      - Thresholded: split_group contains threshold_* (or easy/medium/hard) subdirs.
    """
    jobs = []
    for split_group in expand_split_groups(value_root, split_groups):
        split_root = value_root / split_group
        if not split_root.exists():
            continue

        # Direct layout — files live right in the split group dir
        if _find_split_file(split_root, "train") and not explicit_thresholds:
            split_name = "random" if is_random_split_group(split_group) else split_group
            jobs.append((split_group, split_name, split_root))
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


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def get_split_meta(train_csv: Path, val_csv: Path, test_csv: Path, ratio_tolerance: float):
    train_size = len(_read_table(train_csv))
    val_size = len(_read_table(val_csv))
    test_size = len(_read_table(test_csv))
    total = train_size + val_size + test_size
    if total == 0:
        train_ratio = val_ratio = test_ratio = 0.0
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
        "train_size": train_size, "val_size": val_size, "test_size": test_size,
        "train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio,
        "small_split_flag": small_split_flag,
    }


# ---------------------------------------------------------------------------
# Hparams loading
# ---------------------------------------------------------------------------

def load_hparams(args, parser):
    hparams_path = Path(args.hparams_json) if args.hparams_json else DEFAULT_HPARAMS_JSON
    if not hparams_path.exists():
        return
    with open(hparams_path) as f:
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


# ---------------------------------------------------------------------------
# Feature building (sequential — shared cache is not concurrent-safe)
# ---------------------------------------------------------------------------

def build_features(csv_path: Path, out_pkl: Path, args, gpu_id: str):
    if out_pkl.exists() and not args.overwrite:
        return
    cmd = [
        sys.executable, str(BUILD_SCRIPT),
        "--input_csv", str(csv_path),
        "--output_pkl", str(out_pkl),
        "--target_col", args.target_col,
        "--sequence_col", args.sequence_col,
        "--smiles_col", args.smiles_col,
        "--prot_batch_size", str(args.prot_batch_size),
        "--mol_batch_size", str(args.mol_batch_size),
        "--cache_dir", args.cache_dir,
    ]
    if args.no_cache_read:
        cmd.append("--no_cache_read")
    if args.no_cache_write:
        cmd.append("--no_cache_write")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


# ---------------------------------------------------------------------------
# Training (one subprocess per experiment, called from a worker thread)
# ---------------------------------------------------------------------------

def training_command(exp, args, gpu_id: str):
    return [
        sys.executable, str(TRAIN_SCRIPT),
        "--train_pkl", str(exp["train_pkl"]),
        "--val_pkl", str(exp["val_pkl"]),
        "--test_pkl", str(exp["test_pkl"]),
        "--out_dir", str(exp["run_dir"]),
        "--task_name", exp["task_name"],
        "--batch_size", str(args.train_batch_size),
        "--lr", str(args.lr),
        "--drop_rate", str(args.drop_rate),
        "--epochs", str(args.epochs),
        "--device", "cuda:0",  # subprocess sees only one GPU via CUDA_VISIBLE_DEVICES
        "--patience", str(args.patience),
        "--min_delta", str(args.min_delta),
        "--seed", str(exp["seed"]),
        *(["--skip_singleton_batch"] if args.skip_singleton_batch else []),
        *(["--no_early_stopping"] if args.no_early_stopping else []),
    ]


def run_experiment(exp, args, gpu_id: str):
    exp["run_dir"].mkdir(parents=True, exist_ok=True)
    metric_path = exp["run_dir"] / "final_results_test.csv"
    if metric_path.exists() and not args.overwrite:
        return {"status": "skipped", **_exp_meta(exp, gpu_id)}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    cmd = training_command(exp, args, gpu_id)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    return {"status": "completed", **_exp_meta(exp, gpu_id)}


def _exp_meta(exp, gpu_id):
    return {
        "gpu_id": gpu_id,
        "split_group": exp["split_group"],
        "threshold": exp["threshold"],
        "seed": exp["seed"],
        "run_dir": str(exp["run_dir"]),
    }


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------

def run_parallel(experiments, args):
    work_queue = queue.Queue()
    for exp in experiments:
        work_queue.put(exp)

    results = []
    result_lock = threading.Lock()

    def worker(gpu_id, slot):
        while True:
            try:
                exp = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_experiment(exp, args, gpu_id)
                result["slot"] = slot
            except Exception as exc:
                result = {
                    "status": "failed",
                    "error": str(exc),
                    "slot": slot,
                    **_exp_meta(exp, gpu_id),
                }
            with result_lock:
                results.append(result)
            work_queue.task_done()

    threads = []
    for gpu_id in args.gpus:
        for slot in range(args.runs_per_gpu):
            t = threading.Thread(target=worker, args=(str(gpu_id), slot), daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    return results


# ---------------------------------------------------------------------------
# Summary aggregation (mirrors run_split_benchmarks.py)
# ---------------------------------------------------------------------------

def write_summaries(run_rows, value_root: Path, args):
    runs_df = pd.DataFrame(run_rows)
    runs_df["threshold_num"] = runs_df["threshold"].map(_threshold_to_float)
    runs_df = runs_df.sort_values(["split_group", "threshold_num", "seed"]).drop(columns=["threshold_num"])

    runs_path = value_root / "catapro_summary_runs.csv"
    runs_df.to_csv(runs_path, index=False)

    for split_group, g in runs_df.groupby("split_group", sort=False):
        g.to_csv(value_root / f"catapro_summary_runs__{_slug(split_group)}.csv", index=False)

    metric_cols = [c for c in ["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"] if c in runs_df.columns]
    group_cols = ["value_type", "split_group", "threshold"]

    threshold_rows = []
    for keys, g in runs_df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(g["seed"].nunique())
        for c in ["train_size", "val_size", "test_size", "train_ratio", "val_ratio", "test_ratio", "small_split_flag"]:
            if c in g.columns:
                row[c] = g[c].iloc[0]
        for m in metric_cols:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_var"] = float(g[m].var(ddof=1)) if len(g) > 1 else 0.0
        threshold_rows.append(row)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df["threshold_num"] = threshold_df["threshold"].map(_threshold_to_float)
    threshold_df = threshold_df.sort_values(["split_group", "threshold_num"]).drop(columns=["threshold_num"])
    threshold_df.to_csv(value_root / "catapro_summary_thresholds.csv", index=False)
    threshold_df.to_csv(value_root / "catapro_summary.csv", index=False)

    for split_group, g_th in threshold_df.groupby("split_group", sort=False):
        slug = _slug(split_group)
        g_th.to_csv(value_root / f"catapro_summary_thresholds__{slug}.csv", index=False)
        g_th.to_csv(value_root / f"catapro_summary__{slug}.csv", index=False)

    by_split_rows = []
    for split_group, g in threshold_df.groupby("split_group", sort=False):
        row = {"value_type": args.value_type, "split_group": split_group, "n_thresholds": len(g)}
        for m in metric_cols:
            row[f"{m}_mean_over_thresholds"] = float(g[f"{m}_mean"].mean())
            row[f"{m}_var_over_thresholds"] = float(g[f"{m}_mean"].var(ddof=1)) if len(g) > 1 else 0.0
        by_split_rows.append(row)
    pd.DataFrame(by_split_rows).to_csv(value_root / "catapro_summary_by_split_group.csv", index=False)

    metric_key = f"{args.primary_metric}_mean"
    if metric_key in threshold_df.columns:
        ranked_df = threshold_df.sort_values(metric_key, ascending=not args.higher_is_better)
        ranked_df.to_csv(value_root / "catapro_summary_ranked.csv", index=False)
        for split_group, g_rank in ranked_df.groupby("split_group", sort=False):
            g_rank.to_csv(value_root / f"catapro_summary_ranked__{_slug(split_group)}.csv", index=False)

    print(f"Summaries written to {value_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallel CataPro benchmark: build features sequentially, train across GPUs in parallel."
    )

    # ---- parallelism ----
    parser.add_argument("--gpus", nargs="+", required=True,
                        help="GPU IDs to use (e.g. 0 1 2 3).")
    parser.add_argument("--runs_per_gpu", type=int, default=1,
                        help="Concurrent training jobs per GPU.")
    parser.add_argument("--feature_gpu", type=str, default=None,
                        help="GPU ID for Phase 1 feature building (default: first GPU in --gpus).")

    # ---- data ----
    parser.add_argument("--base_dir",
                        default="/home/adhil/github/EMULaToR/data/processed/baselines/CataPro",
                        type=str)
    parser.add_argument("--value_type", default=None, choices=["kcat", "km", "ki"], type=str,
                        help="Value type subdirectory under base_dir. Omit if using --value_root.")
    parser.add_argument("--value_root", nargs="+", default=None,
                        help="One or more direct paths to split-group directories (bypasses base_dir/value_type).")
    parser.add_argument("--split_groups", nargs="+",
                        default=["enzyme_sequence_splits", "substrate_splits"])
    parser.add_argument("--thresholds", nargs="+", default=None)

    # ---- columns ----
    parser.add_argument("--target_col", default="log10_value", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)

    # ---- feature building ----
    parser.add_argument("--prot_batch_size", default=32, type=int)
    parser.add_argument("--mol_batch_size", default=64, type=int)
    parser.add_argument("--cache_dir", default="emulator_bench/.cache_embeddings", type=str)
    parser.add_argument("--no_cache_read", action="store_true")
    parser.add_argument("--no_cache_write", action="store_true")

    # ---- training hparams (overrideable via JSON) ----
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--drop_rate", default=0.0, type=float)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--min_delta", default=0.001, type=float)
    parser.add_argument("--skip_singleton_batch", action="store_true")
    parser.add_argument("--no_early_stopping", "--no-early-stopping", action="store_true",
                        help="Disable early stopping and train for the full --epochs.")
    parser.add_argument("--hparams_json", type=str, default=None)

    # ---- experiment control ----
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--ratio_tolerance", type=float, default=0.02)
    parser.add_argument("--primary_metric", type=str, default="MSE",
                        choices=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"])
    parser.add_argument("--higher_is_better", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    if args.runs_per_gpu < 1:
        raise ValueError("--runs_per_gpu must be >= 1")

    load_hparams(args, parser)

    # Resolve all value roots
    value_roots = []
    if args.value_root:
        for vr in args.value_root:
            value_roots.append(Path(vr).expanduser())
    elif args.value_type:
        value_roots.append(Path(args.base_dir).expanduser() / args.value_type)
    else:
        raise ValueError("Provide --value_type or --value_root.")

    for vr in value_roots:
        if not vr.exists():
            raise FileNotFoundError(f"Value root directory not found: {vr}")

    feature_gpu = str(args.feature_gpu) if args.feature_gpu is not None else str(args.gpus[0])

    # Collect all jobs across all value roots
    all_root_jobs = {}  # value_root -> list of raw (split_group, split_name, split_dir)
    for vr in value_roots:
        jobs = discover_split_jobs(vr, args.split_groups, args.thresholds)
        all_root_jobs[vr] = jobs

    import math
    total_splits = sum(len(j) for j in all_root_jobs.values())
    total_runs = total_splits * len(args.seeds)
    n_slots = len(args.gpus) * args.runs_per_gpu
    n_rounds = math.ceil(total_runs / n_slots) if n_slots > 0 else total_runs

    print(f"Value roots    : {[vr.name for vr in value_roots]}")
    print(f"Seeds          : {args.seeds}  ({len(args.seeds)} seeds)")
    print(f"Splits         : {total_splits}  ({' + '.join(str(len(all_root_jobs[vr])) for vr in value_roots)} across roots)")
    print(f"Total runs     : {total_splits} splits × {len(args.seeds)} seeds = {total_runs}")
    print(f"Concurrency    : {len(args.gpus)} GPU(s) × {args.runs_per_gpu} slots = {n_slots} concurrent runs")
    print(f"Estimated rounds: ~{n_rounds}")

    if args.dry_run:
        run_index = 0
        for vr, jobs in all_root_jobs.items():
            root_runs = len(jobs) * len(args.seeds)
            print(f"\n{vr.name}:  ({len(jobs)} splits × {len(args.seeds)} seeds = {root_runs} runs)")
            for split_group, split_name, _ in jobs:
                for seed in args.seeds:
                    run_index += 1
                    print(f"  [{run_index:3d}/{total_runs}]  {split_group}/{split_name}  seed={seed}")
        return

    # ------------------------------------------------------------------
    # Phase 1: build features sequentially across all roots
    # ------------------------------------------------------------------
    print(f"\n=== Phase 1: Building features (CUDA_VISIBLE_DEVICES={feature_gpu}) ===")
    # per_root_valid_jobs maps value_root -> list of enriched job dicts
    per_root_valid_jobs = {vr: [] for vr in value_roots}
    all_experiments = []

    for vr in value_roots:
        value_type = vr.name
        jobs = all_root_jobs[vr]
        if not jobs:
            print(f"[skip] no splits found in {vr}")
            continue

        for split_group, split_name, split_dir in tqdm(jobs, desc=f"{value_type} features", unit="split"):
            train_csv = _find_split_file(split_dir, "train")
            val_csv = _find_split_file(split_dir, "val")
            test_csv = _find_split_file(split_dir, "test")
            if not (train_csv and val_csv and test_csv):
                print(f"[skip] missing train/val/test files in {split_dir}")
                continue

            feats_dir = split_dir / "catapro_features"
            feats_dir.mkdir(parents=True, exist_ok=True)

            train_pkl = feats_dir / "train_feats.pkl"
            val_pkl = feats_dir / "val_feats.pkl"
            test_pkl = feats_dir / "test_feats.pkl"

            build_features(train_csv, train_pkl, args, feature_gpu)
            build_features(val_csv, val_pkl, args, feature_gpu)
            build_features(test_csv, test_pkl, args, feature_gpu)

            job_dict = {
                "value_root": vr,
                "value_type": value_type,
                "split_group": split_group,
                "threshold": split_name,
                "split_dir": split_dir,
                "train_csv": train_csv,
                "val_csv": val_csv,
                "test_csv": test_csv,
                "train_pkl": train_pkl,
                "val_pkl": val_pkl,
                "test_pkl": test_pkl,
            }
            per_root_valid_jobs[vr].append(job_dict)

            results_dir = split_dir / "catapro_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            for seed in args.seeds:
                all_experiments.append({
                    **job_dict,
                    "seed": seed,
                    "run_dir": results_dir / f"seed_{seed}",
                    "task_name": f"{value_type}_{split_group}_{split_name}_seed{seed}",
                })

    if not all_experiments:
        print("No valid experiments after feature building. Exiting.")
        return

    # ------------------------------------------------------------------
    # Phase 2: dispatch all training experiments across GPUs in parallel
    # ------------------------------------------------------------------
    print(f"\n=== Phase 2: Parallel training ({len(all_experiments)} experiments) ===")
    dispatch_results = run_parallel(all_experiments, args)

    failed = [r for r in dispatch_results if r["status"] == "failed"]
    if failed:
        print(f"\n[WARNING] {len(failed)} experiment(s) failed:")
        for r in failed:
            print(f"  {r['split_group']}/{r['threshold']} seed={r['seed']}: {r.get('error', '?')}")

    # ------------------------------------------------------------------
    # Phase 3: collect results and write per-root summaries
    # ------------------------------------------------------------------
    print("\n=== Phase 3: Aggregating results ===")
    for vr in value_roots:
        value_type = vr.name
        valid_jobs = per_root_valid_jobs[vr]
        run_rows = []
        for job in valid_jobs:
            for seed in args.seeds:
                run_dir = job["split_dir"] / "catapro_results" / f"seed_{seed}"
                final_test_csv = run_dir / "final_results_test.csv"
                if not final_test_csv.exists():
                    continue
                row = pd.read_csv(final_test_csv).iloc[0].to_dict()
                row["value_type"] = value_type
                row["split_group"] = job["split_group"]
                row["threshold"] = job["threshold"]
                row["seed"] = seed
                row["results_dir"] = str(run_dir)
                row.update(get_split_meta(job["train_csv"], job["val_csv"], job["test_csv"], args.ratio_tolerance))
                run_rows.append(row)
        if run_rows:
            # patch args.value_type for write_summaries labeling
            args.value_type = value_type
            write_summaries(run_rows, vr, args)
        else:
            print(f"No completed experiments for {value_type}.")


if __name__ == "__main__":
    main()
