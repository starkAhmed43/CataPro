import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import hashlib
import os
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys


PROT_MODEL_ID = "Rostlab/prot_t5_xl_uniref50"
MOL_MODEL_ID = "laituan245/molt5-base-smiles2caption"
CACHE_VERSION = "v1"


def _ensure_cache_root(cache_dir):
    if cache_dir is None:
        return None
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _cache_key(namespace, value):
    text = f"{CACHE_VERSION}|{namespace}|{value}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_file(cache_root, namespace, key):
    subdir = cache_root / namespace
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{key}.npy"


def _load_cache_vec(cache_root, namespace, key):
    fpath = _cache_file(cache_root, namespace, key)
    if not fpath.exists():
        return None
    try:
        return np.load(fpath, allow_pickle=False)
    except Exception:
        return None


def _save_cache_vec(cache_root, namespace, key, vec):
    fpath = _cache_file(cache_root, namespace, key)
    tmp = fpath.with_suffix(f".tmp.{os.getpid()}.npy")
    np.save(tmp, np.asarray(vec, dtype=np.float32))
    os.replace(tmp, fpath)


def Seq_to_vec(Sequence, batch_size=8, cache_dir=None, cache_read=True, cache_write=True):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    cache_root = _ensure_cache_root(cache_dir)

    processed = []
    for seq in Sequence:
        seq = str(seq)
        if len(seq) > 1000:
            seq = seq[:500] + seq[-500:]
        seq = re.sub(r"[UZOB]", "X", seq)
        processed.append(" ".join(list(seq)))

    if len(processed) == 0:
        return np.zeros((0, 1024), dtype=np.float32)

    seq_to_indices = {}
    for i, seq in enumerate(processed):
        seq_to_indices.setdefault(seq, []).append(i)

    unique_sequences = list(seq_to_indices.keys())
    unique_sequences.sort(key=len, reverse=True)
    seq_to_embedding = {}

    misses = []
    for seq in unique_sequences:
        if cache_root is not None and cache_read:
            key = _cache_key(PROT_MODEL_ID, seq)
            cached = _load_cache_vec(cache_root, "prot_t5", key)
            if cached is not None:
                seq_to_embedding[seq] = cached.astype(np.float32)
                continue
        misses.append(seq)

    if len(misses) > 0:
        tokenizer = T5Tokenizer.from_pretrained(PROT_MODEL_ID, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(PROT_MODEL_ID)
        gc.collect()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model = model.eval()

        for start in tqdm(range(0, len(misses), batch_size), desc="ProtT5 embedding", unit="batch"):
            batch_sequences = misses[start:start + batch_size]
            ids = tokenizer.batch_encode_plus(batch_sequences, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            with torch.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            for seq_num in range(embedding.shape[0]):
                seq_len = int((attention_mask[seq_num] == 1).sum().item())
                seq_emd = embedding[seq_num][:seq_len - 1]
                vec = seq_emd.mean(dim=0).cpu().numpy().astype(np.float32)
                seq = batch_sequences[seq_num]
                seq_to_embedding[seq] = vec
                if cache_root is not None and cache_write:
                    key = _cache_key(PROT_MODEL_ID, seq)
                    _save_cache_vec(cache_root, "prot_t5", key, vec)

    features_normalize = np.zeros((len(processed), len(next(iter(seq_to_embedding.values())))), dtype=np.float32)
    for seq, indices in seq_to_indices.items():
        features_normalize[indices, :] = seq_to_embedding[seq]

    return features_normalize


def GetMACCSKeys(smiles_list, cache_dir=None, cache_read=True, cache_write=True):
    """
    Output: np.array, size is 167.
    """
    cache_root = _ensure_cache_root(cache_dir)

    if len(smiles_list) == 0:
        return np.zeros((0, 167), dtype=np.float32)

    smiles_to_indices = {}
    for i, smile in enumerate(smiles_list):
        smiles_to_indices.setdefault(str(smile), []).append(i)

    unique_smiles = list(smiles_to_indices.keys())
    unique_smiles.sort(key=len, reverse=True)

    smiles_to_fp = {}
    misses = []
    for smile in unique_smiles:
        if cache_root is not None and cache_read:
            key = _cache_key("maccs_167", smile)
            cached = _load_cache_vec(cache_root, "maccs", key)
            if cached is not None:
                smiles_to_fp[smile] = cached.astype(np.float32)
                continue
        misses.append(smile)

    for smile in tqdm(misses, desc="MACCS fingerprint", unit="smi"):
        mol = Chem.MolFromSmiles(smile)
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_str = fp.ToBitString()
        fp_array = np.array([int(i) for i in fp_str], dtype=np.float32)
        smiles_to_fp[smile] = fp_array
        if cache_root is not None and cache_write:
            key = _cache_key("maccs_167", smile)
            _save_cache_vec(cache_root, "maccs", key, fp_array)

    final_values = np.zeros((len(smiles_list), len(next(iter(smiles_to_fp.values())))), dtype=np.float32)
    for smile, indices in smiles_to_indices.items():
        final_values[indices, :] = smiles_to_fp[smile]

    return final_values


def get_molT5_embed(smiles_list, batch_size=16, cache_dir=None, cache_read=True, cache_write=True):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    cache_root = _ensure_cache_root(cache_dir)

    if len(smiles_list) == 0:
        return np.zeros((0, 768), dtype=np.float32)

    smiles_to_indices = {}
    for i, smile in enumerate(smiles_list):
        smiles_to_indices.setdefault(str(smile), []).append(i)

    unique_smiles = list(smiles_to_indices.keys())
    unique_smiles.sort(key=len, reverse=True)
    smiles_to_embedding = {}

    misses = []
    for smile in unique_smiles:
        if cache_root is not None and cache_read:
            key = _cache_key(MOL_MODEL_ID, smile)
            cached = _load_cache_vec(cache_root, "mol_t5", key)
            if cached is not None:
                smiles_to_embedding[smile] = cached.astype(np.float32)
                continue
        misses.append(smile)

    if len(misses) > 0:
        tokenizer = T5Tokenizer.from_pretrained(MOL_MODEL_ID)
        model = T5EncoderModel.from_pretrained(MOL_MODEL_ID)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model = model.eval()

        for start in tqdm(range(0, len(misses), batch_size), desc="MolT5 embedding", unit="batch"):
            batch_smiles = misses[start:start + batch_size]
            encoded = tokenizer(batch_smiles, return_tensors="pt", padding="longest")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                last_hidden_states = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            for i in range(last_hidden_states.shape[0]):
                seq_len = int((attention_mask[i] == 1).sum().item())
                embed = last_hidden_states[i][:seq_len - 1].mean(dim=0).detach().cpu().numpy().astype(np.float32)
                smile = batch_smiles[i]
                smiles_to_embedding[smile] = embed
                if cache_root is not None and cache_write:
                    key = _cache_key(MOL_MODEL_ID, smile)
                    _save_cache_vec(cache_root, "mol_t5", key, embed)

    final_values = np.zeros((len(smiles_list), len(next(iter(smiles_to_embedding.values())))), dtype=np.float32)
    for smile, indices in smiles_to_indices.items():
        final_values[indices, :] = smiles_to_embedding[smile]

    return final_values
