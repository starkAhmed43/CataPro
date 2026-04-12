import argparse

import numpy as np
import pandas as pd
import torch as th

from model import SingleTaskRegressor


def predict_from_pkl(input_pkl, ckpt_path, out_csv, device="cuda:0", drop_rate=0.0):
    df = pd.read_pickle(input_pkl)

    # If a label exists in the last column, drop it before inference.
    feature_values = df.values
    if feature_values.shape[1] == 1960:
        feature_values = feature_values[:, :-1]

    x = th.from_numpy(feature_values.astype(np.float32)).to(device)

    model = SingleTaskRegressor(drop_rate=drop_rate, device=device)
    model.load_state_dict(th.load(ckpt_path, map_location=device))
    model.eval()

    with th.no_grad():
        ezy_feats = x[:, :1024]
        sbt_feats = x[:, 1024:]
        pred = model(ezy_feats, sbt_feats).cpu().numpy().reshape(-1, 1)

    out_df = pd.DataFrame(pred, index=df.index, columns=["pred"])
    out_df.to_csv(out_csv, float_format="%.6f")
    print(f"Saved predictions to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Predict with unified single-target architecture using a task-specific ckpt.")
    parser.add_argument("--input_pkl", required=True, type=str, help="Feature pickle with 1959 feature columns, optional trailing label.")
    parser.add_argument("--ckpt_path", required=True, type=str, help="Checkpoint path trained by train_single_target_tvt.py.")
    parser.add_argument("--out_csv", required=True, type=str, help="Output CSV path for predictions.")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--drop_rate", default=0.0, type=float)
    args = parser.parse_args()

    predict_from_pkl(args.input_pkl, args.ckpt_path, args.out_csv, args.device, args.drop_rate)


if __name__ == "__main__":
    main()
