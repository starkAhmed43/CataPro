# emulator_bench

This folder adds a strict train/val/test (TVT) workflow for benchmarking CataPro on custom dissimilarity-based splits while keeping OG CataPro code unchanged.

## Unified no-duplication path

Use this path for one architecture code + multiple task-specific checkpoints.

- model.py: single task-agnostic architecture (same as OG kcat/Km model family)
- utils.py: shared training/eval/metrics helpers
- train_single_target_tvt.py: single trainer for kcat, Km, or Ki
- predict_single_target.py: single inference script, only ckpt changes per target type

With this setup, the code is shared and only checkpoint files differ per endpoint.

## What stays OG

- Original model architectures are reused from training/kcat, training/Km, and training/kcat_over_Km.
- Original loss/metrics/training-epoch utilities are reused from each task's utils.py.
- Original 10-fold scripts are not modified.

Only split logic is changed from CV folds to explicit TVT files.

## Raw CSV schema expected by feature builder

The feature script reads CSV files with index_col=0, so keep a stable first index column.

Required columns for all tasks:

- Sequence
- Smiles

Task-specific target columns (defaults, can be overridden with args):

- single_target task: target column passed by --target_col (default: Ki)
- kcat task: kcat(s^-1)
- km task: Km(M)
- kcat_over_km task: kcat(s^-1) and Km(M)

## Step 1: Build TVT feature pickles

Use your mldb conda env when running Python:

conda run -n mldb python emulator_bench/build_tvt_features.py --input_csv path/to/train.csv --output_pkl path/to/train_ki.pkl --target_col Ki

Repeat for val and test, and for each task.

### Persistent cache (recommended for multiple split benchmarks)

The feature builder now caches per-item vectors to disk and reuses them across runs/splits:

- ProtT5 sequence embeddings
- MolT5 smiles embeddings
- MACCS fingerprints

Default cache directory:

- `emulator_bench/.cache_embeddings`

Useful flags:

- `--cache_dir <path>`: custom cache directory
- `--no_cache_read`: force recompute without reading cache
- `--no_cache_write`: do not persist new vectors

Example:

conda run -n mldb python emulator_bench/build_tvt_features.py --input_csv path/to/train.csv --output_pkl path/to/train_ki.pkl --target_col Ki --cache_dir /fastssd/catapro_cache

Output pickle layout follows OG tensor slicing assumptions:

- First 1959 columns: concatenated features (1024 ProtT5 + 768 MolT5 + 167 MACCS)
- Tail label columns:
  - single_target: one label column at the end
  - kcat: one label column at the end
  - km: one label column at the end
  - kcat_over_km: two label columns at the end in order [kcat, Km]

## Step 2: Train a single-target model (kcat, Km, or Ki)

Ki example:

conda run -n mldb python emulator_bench/train_single_target_tvt.py --train_pkl path/to/train_ki.pkl --val_pkl path/to/val_ki.pkl --test_pkl path/to/test_ki.pkl --out_dir path/to/out_ki --task_name ki --device cuda:0

kcat example:

conda run -n mldb python emulator_bench/train_single_target_tvt.py --train_pkl path/to/train_kcat.pkl --val_pkl path/to/val_kcat.pkl --test_pkl path/to/test_kcat.pkl --out_dir path/to/out_kcat --task_name kcat --device cuda:0

Km example:

conda run -n mldb python emulator_bench/train_single_target_tvt.py --train_pkl path/to/train_km.pkl --val_pkl path/to/val_km.pkl --test_pkl path/to/test_km.pkl --out_dir path/to/out_km --task_name km --device cuda:0 --skip_singleton_batch

## Step 3: Predict with one shared inference script

conda run -n mldb python emulator_bench/predict_single_target.py --input_pkl path/to/test_ki.pkl --ckpt_path path/to/out_ki/bestmodel.pth --out_csv path/to/ki_predictions.csv --device cuda:0

Switch only --ckpt_path to use kcat or Km checkpoint with the same script.

## Optional: kcat_over_km benchmark path

If you still want the original activity-style head, use the dedicated script:

conda run -n mldb python emulator_bench/train_kcat_over_km_tvt.py --train_pkl path/to/train_act.pkl --val_pkl path/to/val_act.pkl --test_pkl path/to/test_act.pkl --kcat_model_path path/to/out_kcat/bestmodel.pth --km_model_path path/to/out_km/bestmodel.pth --out_dir path/to/out_act --device cuda:0

Legacy split-specific scripts are retained for comparison:

- train_kcat_tvt.py
- train_km_tvt.py

## Outputs in each out_dir

- bestmodel.pth
- logfile.csv
- time_running.dat
- results_val.csv
- results_test.csv
- pred_label_val.csv
- pred_label_test.csv
- final_results_test.csv

`results_val.csv` and `results_test.csv` contain:

- PCC
- SCC
- R2
- RMSE
- MSE
- MAE
- loss

`final_results_test.csv` contains:

- PCC
- SCC
- R2
- RMSE
- MSE
- MAE

## Notes for fair benchmarking

- Keep your dissimilarity split fixed outside this repo.
- Do not mix val/test records into train pickles.
- Report metrics from final_results_test.csv for your benchmark table.

## Multi-split benchmark runner

To run all thresholds across both split families for one value type (`kcat`, `km`, or `ki`), use:

CUDA_VISIBLE_DEVICES=0 conda run -n mldb python emulator_bench/run_split_benchmarks.py --value_type kcat --device cuda:0 --seeds 0 1 2 3 4 --primary_metric MSE

This script will, for each discovered threshold directory:

- build train/val/test features with cache reuse
- train from scratch on that split for each seed
- save outputs under each threshold directory:
  - `catapro_features/train_feats.pkl`
  - `catapro_features/val_feats.pkl`
  - `catapro_features/test_feats.pkl`
  - `catapro_results/seed_<seed>/*`

It also computes a `small_split_flag` based on deviation from the target 80:10:10 split ratio (train:val:test), without skipping any split.

It writes summaries at:

- `<base_dir>/<value_type>/catapro_summary_runs.csv` (one row per seed-run)
- `<base_dir>/<value_type>/catapro_summary_thresholds.csv` (mean/variance across seeds)
- `<base_dir>/<value_type>/catapro_summary_by_split_group.csv` (enzyme vs substrate aggregates)
- `<base_dir>/<value_type>/catapro_summary_ranked.csv` (ranked by primary metric)
- `<base_dir>/<value_type>/catapro_summary.csv` (backward-compatible threshold summary)

Useful options:

- `--base_dir /home/adhil/github/EMULaToR/data/processed/baselines/CataPro`
- `--split_groups enzyme_sequence_splits substrate_splits`
- `--thresholds threshold_0.1 threshold_0.2`
- `--cache_dir /path/to/shared_cache`
- `--seeds 0 1 2 3 4`
- `--ratio_tolerance 0.02`
- `--primary_metric MSE`
- `--dry_run`

You can also load tuned hyperparameters from JSON:

- `--hparams_json <path/to/best_hparams.json>`

## Optuna hyperparameter tuning

Install Optuna in your env if needed:

conda run -n mldb pip install optuna

Tune hyperparameters on a representative subset of split/threshold jobs:

CUDA_VISIBLE_DEVICES=0 conda run -n mldb python emulator_bench/tune_optuna.py --value_type ki --device cuda:0 --metric MSE --eval_split val --seeds 0 1 2 --max_jobs 6 --n_trials 40

This will:

- reuse/build `catapro_features/*_feats.pkl`
- run Optuna trials over:
  - `train_batch_size`
  - `lr`
  - `drop_rate`
  - `patience`
  - `min_delta`
- optimize mean metric across all selected jobs and seeds
- persist results under:
  - `<base_dir>/<value_type>/optuna_studies/*_best_hparams.json`
  - `<base_dir>/<value_type>/optuna_studies/*_trials.csv`

By default, the tuner now uses an in-memory Optuna study (no SQLite DB).
If you want persistent Optuna storage, pass `--storage` explicitly (for example, a sqlite URL).

Then run your full benchmark with frozen tuned params:

CUDA_VISIBLE_DEVICES=0 conda run -n mldb python emulator_bench/run_split_benchmarks.py --value_type ki --device cuda:0 --seeds 0 1 2 3 4 --primary_metric MSE --hparams_json /home/adhil/github/EMULaToR/data/processed/baselines/CataPro/ki/optuna_studies/catapro_ki_mse_best_hparams.json
