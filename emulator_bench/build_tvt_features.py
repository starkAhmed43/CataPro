import argparse

import numpy as np
import pandas as pd

from feature_utils import Seq_to_vec, GetMACCSKeys, get_molT5_embed


def _require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_features(
    input_csv,
    output_pkl,
    target_col,
    sequence_col="sequence",
    smiles_col="smiles",
    prot_batch_size=8,
    mol_batch_size=16,
    cache_dir="emulator_bench/.cache_embeddings",
    cache_read=True,
    cache_write=True,
):
    p = str(input_csv)
    df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)

    _require_columns(df, [sequence_col, smiles_col])

    seq_list = df[sequence_col].astype(str).tolist()
    smi_list = df[smiles_col].astype(str).tolist()

    prot_feats = Seq_to_vec(
        seq_list,
        batch_size=prot_batch_size,
        cache_dir=cache_dir,
        cache_read=cache_read,
        cache_write=cache_write,
    )
    sbt_molt5_feats = get_molT5_embed(
        smi_list,
        batch_size=mol_batch_size,
        cache_dir=cache_dir,
        cache_read=cache_read,
        cache_write=cache_write,
    )
    sbt_macc = GetMACCSKeys(
        smi_list,
        cache_dir=cache_dir,
        cache_read=cache_read,
        cache_write=cache_write,
    )

    merge_feats = np.concatenate([prot_feats, sbt_molt5_feats, sbt_macc], axis=1)
    out_df = pd.DataFrame(merge_feats, index=df.index)

    _require_columns(df, [target_col])
    out_df["label"] = df[target_col].astype(float).values

    out_df.to_pickle(output_pkl)
    print(f"Saved features to: {output_pkl}")
    print(f"Rows: {len(out_df)} | Columns: {out_df.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TVT feature pickle for single-target tasks (kcat/Km/Ki).")
    parser.add_argument("--input_csv", required=True, type=str, help="Input CSV path.")
    parser.add_argument("--output_pkl", required=True, type=str, help="Output pickle path.")
    parser.add_argument("--sequence_col", default="sequence", type=str, help="CSV column for protein sequence.")
    parser.add_argument("--smiles_col", default="smiles", type=str, help="CSV column for substrate SMILES.")
    parser.add_argument("--target_col", required=True, type=str, help="CSV column used as regression target.")
    parser.add_argument("--prot_batch_size", default=32, type=int, help="Batch size for ProtT5 embedding.")
    parser.add_argument("--mol_batch_size", default=64, type=int, help="Batch size for MolT5 embedding.")
    parser.add_argument(
        "--cache_dir",
        default="emulator_bench/.cache_embeddings",
        type=str,
        help="Directory for persistent embedding/fingerprint cache.",
    )
    parser.add_argument("--no_cache_read", action="store_true", help="Disable reading from cache.")
    parser.add_argument("--no_cache_write", action="store_true", help="Disable writing newly computed vectors to cache.")

    args = parser.parse_args()

    build_features(
        input_csv=args.input_csv,
        output_pkl=args.output_pkl,
        target_col=args.target_col,
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
        prot_batch_size=args.prot_batch_size,
        mol_batch_size=args.mol_batch_size,
        cache_dir=args.cache_dir,
        cache_read=not args.no_cache_read,
        cache_write=not args.no_cache_write,
    )
