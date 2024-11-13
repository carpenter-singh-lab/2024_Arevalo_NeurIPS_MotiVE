import json
import sys
from pathlib import Path

import pandas as pd
import torch

from utils.evaluate import Evaluator, compute_map

arg = sys.argv[1]
config_path = Path(arg)
with config_path.open() as f:
    config = json.load(f)
graph_type = config["graph_type"]
target_type = config["target_type"]
split = config["data_split"]

smap = pd.read_parquet(f"data/{graph_type}/{target_type}/source_map.parquet")[0]
tmap = pd.read_parquet(f"data/{graph_type}/{target_type}/target_map.parquet")[0]
anno = pd.read_parquet(f"data/{graph_type}/{target_type}/{split}/s_t_labels.parquet")
anno = anno.query("subset=='test'")

pivot = config_path.parent / "analysis" / "pivot.csv"
scores = pd.read_csv(pivot).melt("inchi", var_name="genes", value_name="score")
scores["source"] = scores["inchi"].map(smap)
scores["target"] = scores["genes"].map(tmap)
scores.set_index(["source", "target"], inplace=True)
scores["y_true"] = 0
scores.loc[anno.itertuples(index=False), "y_true"] = 1

e = Evaluator("configs/eval/test_evaluation_params.json")
e.config["Robhan_th"] = 0.01
logits = torch.tensor(scores["score"].values)
y_true = torch.tensor(scores["y_true"].values)
edges = torch.tensor(scores.index.to_frame().values.T)
all_metrics = e._robhan(logits, y_true, edges)
ap_scores = compute_map(scores.rename(columns={"inchi": "source", "genes": "target"}))
map_scores = ap_scores.groupby("node_type")["average_precision"].mean()
for k, v in map_scores.items():
    all_metrics[f"mAP_{k}"] = v
all_metrics.update(config)
all_metrics["dataset"] = arg.split("/")[1]
print(all_metrics)
