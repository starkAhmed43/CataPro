import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna
import pandas as pd
from tqdm.auto import tqdm


def discover_threshold_dirs(value_root: Path, split_groups, explicit_thresholds=None):
    jobs = []
    for split_group in split_groups:
        split_root = value_root / split_group
        if not split_root.exists():
            continue

        if explicit_thresholds:
            threshold_dirs = [split_root / t for t in explicit_thresholds]
        else:
            threshold_dirs = [p for p in sorted(split_root.iterdir()) if p.is_dir() and p.name.startswith("threshold_")]

        for threshold_dir in threshold_dirs:
            if threshold_dir.exists():
                jobs.append((split_group, threshold_dir.name, threshold_dir))

    jobs.sort(key=lambda x: (x[0], _threshold_to_float(x[1])))
    return jobs


def ensure_csv_triplet(threshold_dir: Path):
    train_csv = threshold_dir / "train.csv"
    val_csv = threshold_dir / "val.csv"
    test_csv = threshold_dir / "test.csv"
    return train_csv, val_csv, test_csv


def _threshold_to_float(name: str):
    try:
        return float(str(name).split("threshold_")[-1])
    except Exception:
        return float("inf")


def maybe_build_feature(csv_path, out_pkl, args):
    if out_pkl.exists() and not args.overwrite_features:
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


def run_training(train_pkl, val_pkl, test_pkl, out_dir, args, seed, hp, device):
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
        args.task_name,
        "--batch_size",
        str(hp["train_batch_size"]),
        "--lr",
        str(hp["lr"]),
        "--drop_rate",
        str(hp["drop_rate"]),
        "--epochs",
        str(hp["epochs"]),
        "--device",
        device,
        "--patience",
        str(hp["patience"]),
        "--min_delta",
        str(hp["min_delta"]),
        "--seed",
        str(seed),
    ]

    if args.skip_singleton_batch:
        cmd.append("--skip_singleton_batch")

    subprocess.run(cmd, check=True)


def _objective_direction(metric_name: str):
    return "maximize" if metric_name in {"PCC", "SCC", "R2"} else "minimize"


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for CataPro TVT single-target training.")

    parser.add_argument("--base_dir", default="/home/adhil/github/EMULaToR/data/processed/baselines/CataPro", type=str)
    parser.add_argument("--value_type", required=True, choices=["kcat", "km", "ki"], type=str)
    parser.add_argument("--split_groups", nargs="+", default=["enzyme_sequence_splits", "substrate_splits"])
    parser.add_argument(
        "--separate_by_split_group",
        action="store_true",
        help="Run independent Optuna studies for each split group instead of one combined objective.",
    )
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument(
        "--max_jobs",
        type=int,
        default=10,
        help="Use only the first N discovered threshold jobs for tuning (sorted by split + threshold).",
    )

    parser.add_argument("--target_col", default="log10_value", type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)

    parser.add_argument("--prot_batch_size", default=32, type=int)
    parser.add_argument("--mol_batch_size", default=64, type=int)
    parser.add_argument("--cache_dir", default="emulator_bench/.cache_embeddings", type=str)
    parser.add_argument("--no_cache_read", action="store_true")
    parser.add_argument("--no_cache_write", action="store_true")
    parser.add_argument("--overwrite_features", action="store_true")

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--task_name", default="single_target_optuna", type=str)
    parser.add_argument("--skip_singleton_batch", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45], help="Random seeds for training runs within each trial.")

    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None, help="Optional Optuna storage URL. If omitted, uses in-memory study (no DB).")

    parser.add_argument("--metric", type=str, default="MSE", choices=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument(
        "--parallel_runs_per_trial",
        type=int,
        default=1,
        help="Max concurrent train/eval runs inside one trial (across split jobs and seeds).",
    )
    parser.add_argument(
        "--trial_parallelism",
        type=int,
        default=1,
        help="Number of Optuna trials to run concurrently (Optuna n_jobs).",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Optional device list for round-robin assignment when parallelizing runs (e.g. cuda:0 cuda:1).",
    )

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    if args.parallel_runs_per_trial < 1:
        raise ValueError("--parallel_runs_per_trial must be >= 1")
    if args.trial_parallelism < 1:
        raise ValueError("--trial_parallelism must be >= 1")

    base_dir = Path(args.base_dir)
    value_root = base_dir / args.value_type
    if not value_root.exists():
        raise FileNotFoundError(f"Value type directory not found: {value_root}")

    jobs = discover_threshold_dirs(value_root, args.split_groups, args.thresholds)
    if not jobs:
        raise RuntimeError("No threshold jobs discovered. Check --base_dir/--value_type/--split_groups/--thresholds")

    if args.max_jobs and args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not-set>')}")
    print(
        f"Parallelism: parallel_runs_per_trial={args.parallel_runs_per_trial}, "
        f"trial_parallelism={args.trial_parallelism}"
    )
    if args.devices:
        print(f"Device pool: {args.devices}")
    print(f"Tuning jobs ({len(jobs)}):")
    for split_group, threshold_name, threshold_dir in jobs:
        print(f"- {split_group}/{threshold_name}: {threshold_dir}")

    if args.dry_run:
        return

    # Build/reuse features once before optimization.
    feature_progress = tqdm(jobs, desc="Preparing features", unit="job")
    prepared_jobs = []
    for split_group, threshold_name, threshold_dir in feature_progress:
        train_csv, val_csv, test_csv = ensure_csv_triplet(threshold_dir)
        if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
            continue

        feats_dir = threshold_dir / "catapro_features"
        feats_dir.mkdir(parents=True, exist_ok=True)

        train_pkl = feats_dir / "train_feats.pkl"
        val_pkl = feats_dir / "val_feats.pkl"
        test_pkl = feats_dir / "test_feats.pkl"

        maybe_build_feature(train_csv, train_pkl, args)
        maybe_build_feature(val_csv, val_pkl, args)
        maybe_build_feature(test_csv, test_pkl, args)

        prepared_jobs.append((split_group, threshold_name, threshold_dir, train_pkl, val_pkl, test_pkl))

    if not prepared_jobs:
        raise RuntimeError("No valid jobs with train/val/test csv triplets were found.")

    base_study_name = args.study_name or f"catapro_{args.value_type}_{args.metric.lower()}"
    direction = _objective_direction(args.metric)
    artifacts_dir = value_root / "optuna_studies"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run_one_study(study_name, jobs_subset, split_groups_for_metadata):
        sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
        if args.storage:
            study = optuna.create_study(
                study_name=study_name,
                storage=args.storage,
                load_if_exists=True,
                direction=direction,
                sampler=sampler,
            )
        else:
            print("Optuna storage: in-memory (no SQLite DB)")
            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                sampler=sampler,
            )

        run_root = value_root / "catapro_optuna_runs" / study_name
        run_root.mkdir(parents=True, exist_ok=True)

        def _assigned_device(task_idx):
            if args.devices:
                return args.devices[task_idx % len(args.devices)]
            return args.device

        def _run_single(task_idx, split_group, threshold_name, train_pkl, val_pkl, test_pkl, trial_number, seed, hp):
            out_dir = run_root / f"trial_{trial_number}" / split_group / threshold_name / f"seed_{seed}"
            metric_csv = out_dir / f"final_results_{args.eval_split}.csv"

            if not metric_csv.exists():
                run_training(
                    train_pkl,
                    val_pkl,
                    test_pkl,
                    out_dir,
                    args,
                    seed,
                    hp,
                    _assigned_device(task_idx),
                )

            if not metric_csv.exists():
                raise RuntimeError(f"Missing metrics file for trial {trial_number}: {metric_csv}")

            df = pd.read_csv(metric_csv)
            if args.metric not in df.columns:
                raise RuntimeError(f"Metric {args.metric} not found in {metric_csv}")

            return float(df.iloc[0][args.metric])

        def objective(trial: optuna.Trial):
            hp = {
                "train_batch_size": trial.suggest_categorical("train_batch_size", [512, 1024, 2048, 4096]),
                "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
                "drop_rate": trial.suggest_float("drop_rate", 0.0, 0.4),
                "patience": trial.suggest_int("patience", 8, 30),
                "min_delta": trial.suggest_float("min_delta", 1e-5, 5e-3, log=True),
                "epochs": args.epochs,
            }

            tasks = []
            task_idx = 0
            for split_group, threshold_name, _, train_pkl, val_pkl, test_pkl in jobs_subset:
                for seed in args.seeds:
                    tasks.append((task_idx, split_group, threshold_name, train_pkl, val_pkl, test_pkl, trial.number, seed, hp))
                    task_idx += 1

            metric_values = []
            if args.parallel_runs_per_trial == 1:
                for task in tasks:
                    try:
                        metric_values.append(_run_single(*task))
                    except subprocess.CalledProcessError as e:
                        raise optuna.TrialPruned(f"Training failed for trial {trial.number}: {e}")
                    except Exception as e:
                        raise optuna.TrialPruned(str(e))
            else:
                with ThreadPoolExecutor(max_workers=args.parallel_runs_per_trial) as ex:
                    futures = [ex.submit(_run_single, *task) for task in tasks]
                    for f in as_completed(futures):
                        try:
                            metric_values.append(f.result())
                        except subprocess.CalledProcessError as e:
                            raise optuna.TrialPruned(f"Training failed for trial {trial.number}: {e}")
                        except Exception as e:
                            raise optuna.TrialPruned(str(e))

            if not metric_values:
                raise optuna.TrialPruned("No metric values collected.")

            mean_metric = float(sum(metric_values) / len(metric_values))
            trial.set_user_attr("n_runs", len(metric_values))
            trial.set_user_attr("metric", args.metric)
            trial.set_user_attr("eval_split", args.eval_split)
            trial.set_user_attr("mean_metric", mean_metric)
            return mean_metric

        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.trial_parallelism)

        best_hp = dict(study.best_params)
        best_hp["epochs"] = args.epochs

        best_path = artifacts_dir / f"{study_name}_best_hparams.json"
        with open(best_path, "w") as f:
            json.dump(
                {
                    "value_type": args.value_type,
                    "metric": args.metric,
                    "direction": direction,
                    "eval_split": args.eval_split,
                    "seeds": args.seeds,
                    "split_groups": split_groups_for_metadata,
                    "thresholds": args.thresholds,
                    "max_jobs": args.max_jobs,
                    "best_trial_number": study.best_trial.number,
                    "best_value": float(study.best_value),
                    "train_batch_size": int(best_hp["train_batch_size"]),
                    "lr": float(best_hp["lr"]),
                    "drop_rate": float(best_hp["drop_rate"]),
                    "patience": int(best_hp["patience"]),
                    "min_delta": float(best_hp["min_delta"]),
                    "epochs": int(best_hp["epochs"]),
                },
                f,
                indent=2,
            )

        trials_path = artifacts_dir / f"{study_name}_trials.csv"
        study.trials_dataframe().to_csv(trials_path, index=False)

        print(f"[{study_name}] Best trial: {study.best_trial.number}")
        print(f"[{study_name}] Best value ({args.metric}): {study.best_value}")
        print(f"[{study_name}] Best params saved to: {best_path}")
        print(f"[{study_name}] Trial table saved to: {trials_path}")

    if args.separate_by_split_group:
        grouped = {}
        for row in prepared_jobs:
            grouped.setdefault(row[0], []).append(row)

        for split_group, jobs_subset in grouped.items():
            sub_name = f"{base_study_name}__{split_group}"
            print(f"Running separate study for split_group={split_group}: {sub_name}")
            run_one_study(sub_name, jobs_subset, [split_group])
    else:
        run_one_study(base_study_name, prepared_jobs, args.split_groups)


if __name__ == "__main__":
    main()
