import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from copairs import compute
from copairs.map.average_precision import build_rank_lists
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.nn import functional as F


def accuracy(preds_path, acc_path):
    pred = pd.read_parquet(preds_path)
    acc = accuracy_score(pred["y_true"], pred["y_pred"])
    np.array(acc).tofile(acc_path)


def roc_auc(preds_path, auc_path):
    pred = pd.read_parquet(preds_path)
    auc = roc_auc_score(pred["y_true"], pred["score"])
    np.array(auc).tofile(auc_path)


def hits_at_k(preds_path, k, hits_path):
    preds = pd.read_parquet(preds_path)
    logits_neg = preds.query("y_true == 0")["logits"]
    if len(logits_neg) < k:
        hits_k = 1.0
    else:
        kth_neg_score = logits_neg.nlargest(k).iloc[-1]
        logits_pos = preds.query("y_true == 1")["logits"]
        n_pos = len(logits_pos)
        hits_k = (logits_pos >= kth_neg_score).sum() / n_pos
    np.array(hits_k).tofile(hits_path)


def precision_at_k(preds_path, k, pr_path):
    preds = pd.read_parquet(preds_path)
    top_k = preds.nlargest(k, "logits")
    pr_k = top_k["y_true"].mean()
    pr_k.astype(np.float64).tofile(pr_path)


def f1(preds_path, f1_path):
    pred = pd.read_parquet(preds_path)
    f1 = f1_score(pred["y_true"], pred["y_pred"], zero_division=0)
    np.array(f1).tofile(f1_path)


def mrr(preds_path, mrr_path):
    preds = pd.read_parquet(preds_path)
    logits_neg = preds.query("y_true == 0")["logits"].values
    logits_pos = preds.query("y_true == 1")["logits"].values
    logits_pos = logits_pos.reshape(-1, 1)
    optimistic_rank = (logits_neg > logits_pos).sum(axis=1)
    pessimistic_rank = (logits_neg >= logits_pos).sum(axis=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    mrr_list = 1.0 / ranking_list.astype(np.float32)
    mrr = np.mean(mrr_list)
    mrr.astype(np.float64).tofile(mrr_path)


def bce(preds_path, bce_path):
    preds = pd.read_parquet(preds_path)
    logits = torch.tensor(preds["logits"].values)
    y_true = torch.tensor(preds["y_true"].values)
    bce = F.binary_cross_entropy_with_logits(logits, y_true.to(torch.float32))
    bce.cpu().numpy().astype(np.float64).tofile(bce_path)


def average_precision(preds_path, ap_path):
    df = pd.read_parquet(preds_path).reset_index()
    src_ix = df["source"].unique()
    tgt_ix = df["target"].unique()
    num_src, num_tgt = len(src_ix), len(tgt_ix)
    src_mapper = dict(zip(src_ix, range(num_src)))
    tgt_mapper = dict(zip(tgt_ix, range(num_src, num_src + num_tgt)))
    df["src_copairs_id"] = df["source"].map(src_mapper)
    df["tgt_copairs_id"] = df["target"].map(tgt_mapper)
    pos_pairs = df.query("y_true==1")[["src_copairs_id", "tgt_copairs_id"]].values
    neg_pairs = df.query("y_true==0")[["src_copairs_id", "tgt_copairs_id"]].values
    pos_sims = df.query("y_true==1")["score"].values
    neg_sims = df.query("y_true==0")["score"].values
    paired_ix, rel_k_list, counts = build_rank_lists(
        pos_pairs, neg_pairs, pos_sims, neg_sims
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"invalid value encountered in divide")
        ap_scores, null_confs = compute.ap_contiguous(rel_k_list, counts)
    ap_scores = pd.DataFrame(
        {
            "node_id": np.concatenate([src_ix, tgt_ix]),
            "node_type": ["source"] * num_src + ["target"] * num_tgt,
            "average_precision": ap_scores,
            "n_pos_pairs": null_confs[:, 0],
            "n_total_pairs": null_confs[:, 1],
        }
    )
    ap_scores.to_parquet(ap_path)


def mean_average_precision(ap_path, node, map_path):
    ap_scores = pd.read_parquet(ap_path).query(f"node_type=='{node}'")
    map_score = ap_scores["average_precision"].mean()
    np.array(map_score).tofile(map_path)


def success_at_k_ratio(preds_path, node, th, num_path, pct_path):
    """Robhan metric"""
    dframe = pd.read_parquet(preds_path).reset_index()
    dframe["rank"] = dframe.groupby(node)["score"].rank(
        method="min", ascending=False, pct=False
    )
    dframe["rank_pct"] = dframe.groupby(node)["score"].rank(
        method="min", ascending=False, pct=True
    )
    num = dframe.query(f"rank_pct <= {th} and y_true==1")[node].nunique()
    pct = num / dframe[node].nunique()
    np.array(num).tofile(num_path)
    np.array(pct).tofile(pct_path)


def collate(*args, infer_mode):
    *score_paths, config_path, metrics_path = args
    with open(config_path) as f:
        record = json.load(f)
    record["infer_mode"] = infer_mode
    for path in map(Path, score_paths):
        record[path.stem] = np.fromfile(path).item()
    pd.DataFrame([record]).to_parquet(metrics_path)
