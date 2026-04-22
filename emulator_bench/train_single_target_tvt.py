import argparse
import datetime
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model import SingleTaskRegressor
from utils import (
    EarlyStopping,
    evaluate,
    out_results,
    run_a_training_epoch,
    run_an_eval_epoch,
    write_logfile,
)

import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


class TensorDataset(Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)


def build_loader(pkl_path, batch_size, shuffle, workers=8):
    df = pd.read_pickle(pkl_path)
    idx = df.index.tolist()
    values = th.from_numpy(df.values.astype(np.float32))
    ds = TensorDataset(values)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers if shuffle else 0)
    return idx, dl


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_idx, train_loader = build_loader(args.train_pkl, args.batch_size, shuffle=True)
    val_idx, val_loader = build_loader(args.val_pkl, args.batch_size, shuffle=False)
    test_idx, test_loader = build_loader(args.test_pkl, args.batch_size, shuffle=False)

    model = SingleTaskRegressor(drop_rate=args.drop_rate, device=args.device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    stopper = EarlyStopping(args.epochs + 1 if args.no_early_stopping else args.patience, args.min_delta)

    best_model_path = out_dir / "bestmodel.pth"
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record_data = []
    epoch_bar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        train_eval = run_a_training_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            skip_singleton_batch=args.skip_singleton_batch,
        )
        val_pred, val_label, val_eval = run_an_eval_epoch(model, val_loader, args.device)

        epoch_bar.set_postfix(
            train_r2=f"{train_eval[2]:.4f}",
            train_mse=f"{train_eval[4]:.4f}",
            val_r2=f"{val_eval[2]:.4f}",
            val_mse=f"{val_eval[4]:.4f}",
            val_loss=f"{val_eval[-1]:.4f}",
        )

        record_data.append(np.concatenate([np.array([epoch]), train_eval, val_eval], axis=0))
        write_logfile(epoch, record_data, str(out_dir / "logfile.csv"))

        is_best, stop = stopper.check(epoch, val_eval[-1])
        if is_best:
            th.save(model.state_dict().copy(), best_model_path)

        if stop:
            epoch_bar.write("Earlystopping !")
            break

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(out_dir / "time_running.dat", "w") as f:
        f.write(f"Start Time:  {start_time}\\n")
        f.write(f"End Time:  {end_time}\\n")

    model = SingleTaskRegressor(drop_rate=args.drop_rate, device=args.device)
    model.load_state_dict(th.load(best_model_path, map_location=args.device))

    val_pred, val_label, val_eval = run_an_eval_epoch(model, val_loader, args.device)
    test_pred, test_label, test_eval = run_an_eval_epoch(model, test_loader, args.device)

    out_results(val_eval, str(out_dir / "results_val.csv"))
    out_results(test_eval, str(out_dir / "results_test.csv"))

    val_df = pd.DataFrame(
        np.concatenate([val_pred.reshape(-1, 1), val_label.reshape(-1, 1)], axis=1),
        index=val_idx,
        columns=["pred", "label"],
    )
    test_df = pd.DataFrame(
        np.concatenate([test_pred.reshape(-1, 1), test_label.reshape(-1, 1)], axis=1),
        index=test_idx,
        columns=["pred", "label"],
    )
    val_df.to_csv(out_dir / "pred_label_val.csv", float_format="%.4f")
    test_df.to_csv(out_dir / "pred_label_test.csv", float_format="%.4f")

    val_pcc, val_scc, val_r2, val_rmse, val_mse, val_mae = evaluate(val_df["label"].values, val_df["pred"].values)
    val_results = pd.DataFrame(
        np.array([[val_pcc, val_scc, val_r2, val_rmse, val_mse, val_mae]]),
        columns=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"],
    )
    val_results.to_csv(out_dir / "final_results_val.csv", index=False)

    pcc, scc, r2, rmse, mse, mae = evaluate(test_df["label"].values, test_df["pred"].values)
    test_results = pd.DataFrame(np.array([[pcc, scc, r2, rmse, mse, mae]]), columns=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"])
    test_results.to_csv(out_dir / "final_results_test.csv", index=False)

    summary = pd.DataFrame(
        {
            "task_name": [args.task_name],
            "train_size": [len(train_idx)],
            "val_size": [len(val_idx)],
            "test_size": [len(test_idx)],
            "checkpoint": [str(best_model_path)],
        }
    )
    summary.to_csv(out_dir / "run_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-agnostic single-target TVT trainer for kcat/Km/Ki.")
    parser.add_argument("--train_pkl", required=True, type=str)
    parser.add_argument("--val_pkl", required=True, type=str)
    parser.add_argument("--test_pkl", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--task_name", default="single_target", type=str, help="Label for metadata only.")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization and shuffling.")
    parser.add_argument(
        "--skip_singleton_batch",
        action="store_true",
        help="Matches OG Km behavior when a train batch has only one row.",
    )
    parser.add_argument(
        "--no_early_stopping",
        action="store_true",
        help="Disable early stopping and train for the full --epochs.",
    )

    main(parser.parse_args())
