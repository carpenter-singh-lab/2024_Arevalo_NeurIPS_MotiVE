from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from motive import get_all_st_edges, get_loaders, load_graph_helper
from motive.sample_negatives import negative_sampling, select_nodes_to_sample
from utils.evaluate import Evaluator
from mworkflow import init

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mapper = {
    "DOWNREGULATES_CHdG": "downregulates",
    "CdG": "downregulates",
    "UPREGULATES_CHuG": "upregulates",
    "CuG": "upregulates",
    "DRUG_TARGET": "targets",
    "target": "targets",
    "DRUG_CARRIER": "carries",
    "carrier": "carries",
    "DRUG_ENZYME": "enzyme",
    "enzyme": "enzyme",
    "DRUG_TRANSPORTER": "transports",
    "transporter": "transports",
    "BINDS_CHbG": "binds",
    "CbG": "binds",
    "DRUG_BINDING_GENE": "binds",
}


@torch.inference_mode
def run_test_full(model, data, split, th):
    model.eval()
    pos_edges = get_all_st_edges(data)
    pos_edges = torch.tensor(pos_edges)
    edges = data["binds"]["edge_label_index"]
    num_pos = len(data["binds"].edge_label)
    size = num_pos * 10

    source_ix, target_ix = select_nodes_to_sample(data, split)
    neg_edges = negative_sampling(source_ix, target_ix, pos_edges, size)
    ext_edges = torch.cat([edges, neg_edges], axis=1)
    y_true = torch.repeat_interleave(
        torch.tensor([1.0, 0.0]),
        torch.tensor([num_pos, neg_edges.shape[1]]),
    )

    data["binds"]["edge_label_index"] = ext_edges

    data.to(DEVICE)
    logits = model(data)
    scores = logits.cpu().numpy()  # torch.sigmoid(logits).cpu().numpy()

    y_pred = logits > th
    y_true = y_true.to(torch.int32)
    e = Evaluator("configs/eval/test_evaluation_params.json")
    e.config["Robhan_th"] = 0.01
    test_metrics = e.evaluate(logits, y_true, th, data["binds"].edge_label_index)
    data["binds"]["edge_label_index"] = edges

    # save all to results table
    results = pd.DataFrame(ext_edges.T.cpu().numpy(), columns=["source", "target"])
    results["score"] = scores
    results["y_pred"] = y_pred.cpu().numpy()
    results["y_true"] = y_true.cpu().numpy()

    results.sort_values(by=["score"], ascending=False, inplace=True)
    results["percentile"] = results.score.rank(pct=True)
    results.set_index(["source", "target"], inplace=True)

    return results, test_metrics


@torch.inference_mode
def run_test_mini(model, test_loader, th):
    model.eval()
    logits = []
    y_true = []
    src_ids = []
    tgt_ids = []
    edges = []
    for batch in test_loader:
        logits.append(model(batch))
        y_true.append(batch["binds"].edge_label)

        # sampled batch source and target ids of each edge in test set
        test_srcs = batch["binds"].edge_label_index[0]
        test_tgts = batch["binds"].edge_label_index[1]

        # global source and target ids of each edge in test set
        src_ids.append(batch["source"].node_id[test_srcs])
        tgt_ids.append(batch["target"].node_id[test_tgts])
        edges.append(batch["binds"].edge_label_index)

    # save logits, scores, bool predictions, gt, and indices of srcs and tgts
    logits = torch.cat(logits, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred = logits > th
    scores = logits.cpu().numpy()  # torch.sigmoid(logits).cpu().numpy()
    sources = torch.cat(src_ids, dim=0).cpu().numpy()
    targets = torch.cat(tgt_ids, dim=0).cpu().numpy()

    y_true = y_true.to(torch.int32)
    e = Evaluator("configs/eval/test_evaluation_params.json")
    e.config["Robhan_th"] = 0.01
    edges = torch.tensor(np.stack([sources, targets]))
    test_metrics = e.evaluate(logits, y_true, th, edges)

    # save all to results table
    results = pd.DataFrame(sources, columns=["source"])
    results["target"] = targets
    results["score"] = scores
    results["y_pred"] = y_pred.cpu().numpy()
    results["y_true"] = y_true.cpu().numpy()

    results.sort_values(by=["score"], ascending=False, inplace=True)
    results["percentile"] = results.score.rank(pct=True)
    results.set_index(["source", "target"], inplace=True)

    return results, test_metrics


config_path = "test/orf/target/st_expanded/gnn/410618/config.json"
output_path = "test/"
analysis_path = Path(config_path).parent / "analysis"
analysis_path.mkdir(exist_ok=True)

config, model, loaders = init(config_path)
model_path = Path(config_path).parent / "weights.pt"
tgt_type = config["target_type"]
graph_type = config["graph_type"]
leave_out = config["leave_out"]
train_data, valid_data, test_data = load_graph_helper(leave_out, tgt_type, graph_type)
train_loader, val_loader, test_loader = get_loaders(leave_out, tgt_type, graph_type)

best_params = torch.load(model_path, weights_only=True)
best_th = 0
model.load_state_dict(best_params["model_state_dict"])

results_mini, test_metrics_mini = run_test_mini(model, test_loader, best_th)
results_full, test_metrics_full = run_test_full(model, test_data, leave_out, best_th)

joined = results_full.join(results_mini, lsuffix="_full", rsuffix="_mini").dropna()
assert (joined.y_true_full == joined.y_true_mini).all()
pprint((joined.score_full - joined.score_mini).abs().describe())
pprint({"results_mini": test_metrics_mini, "results_full": test_metrics_full})

# Full matrix prediction in test
src = test_data["binds"].edge_label_index[0].unique()
tgt = test_data["binds"].edge_label_index[1].unique()
edges = torch.cartesian_prod(src, tgt).T.contiguous()
gt_edges = test_data["binds"].edge_label_index
y_true = torch.any(torch.all(edges.T[:, None] == gt_edges.T, axis=-1), axis=-1)
with torch.inference_mode():
    model.eval()
    test_data["binds"].edge_label_index = edges
    test_data["binds"].edge_label = y_true
    logits = model(test_data)
    scores = logits  # torch.sigmoid(logits)
    edges = edges.cpu().numpy()
    scores = scores.cpu().numpy()


src_names = pd.read_parquet(f"data/{graph_type}/orf/source.parquet").index
tgt_names = pd.read_parquet(f"data/{graph_type}/orf/target.parquet").index
scores = pd.DataFrame(
    {
        "inchi": src_names[edges[0]],
        "genes": tgt_names[edges[1]],
        "score": scores,
        "y_true": y_true.cpu().numpy(),
    }
)
e = Evaluator("configs/eval/test_evaluation_params.json")
e.config["Robhan_th"] = 0.01
all_metrics = e._robhan(logits, y_true, test_data["binds"].edge_label_index)

pivot = scores.pivot(index="inchi", columns="genes", values="score")
clustergrid = sns.clustermap(pivot, cmap="vlag")
pivot = pivot.iloc[clustergrid.dendrogram_row.reordered_ind]
columns_ord = pivot.columns[clustergrid.dendrogram_col.reordered_ind]
pivot = pivot[columns_ord]
pivot.to_csv(analysis_path / "pivot.csv")
plt.savefig(analysis_path / "clustermap.png", bbox_inches="tight")

ann = pd.read_parquet("./inputs/annotations/compound_gene.parquet")
ann["compound"] = ann["inchikey"].fillna("").str[:14]
results_mini["compound"] = src_names[results_mini.index.get_level_values("source")]
results_mini["gene"] = tgt_names[results_mini.index.get_level_values("target")]
ann_results = results_mini.merge(
    ann, left_on=["compound", "gene"], right_on=["compound", "target"], how="left"
)

scores["rank"] = (
    scores.groupby("genes")["score"]
    .rank(method="min", ascending=False, pct=False)
    .astype(int)
)
scores_ann = scores.query("y_true").merge(
    ann, left_on=["inchi", "genes"], right_on=["compound", "target"], how="left"
)
scores_ann["database"] = scores_ann["database"].astype("category")
scores_ann["database"] = scores_ann["database"].cat.set_categories(
    ["drugrep", "dgidb", "primekg", "hetionet", "biokg", "openbiolink", "pharmebinet"]
)
fig, ax = plt.subplots(figsize=(14, 6))
numsrc = scores["inchi"].nunique()
ax.set_title(f"Rank distribution per relation type. Total: {numsrc}")
scores_ann["rel_type"] = scores_ann["rel_type"].apply(lambda x: mapper.get(x, x))
stats = scores_ann.groupby("rel_type").agg({"rank": ["median", "count"]})
stats.columns = stats.columns.droplevel(0)
stats = stats.sort_values("median")
sns.boxplot(
    data=scores_ann,
    y="rank",
    x="rel_type",
    order=stats.index,
    ax=ax,
    width=1.0,
    gap=0.1,
)

for i, row in enumerate(stats.itertuples()):
    ax.text(
        i,
        row.median,
        f"n={row.count}",
        size="x-small",
        weight="semibold",
        horizontalalignment="center",
        verticalalignment="bottom",
    )
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(analysis_path / "rel_rank.pdf", bbox_inches="tight")
plt.close("all")


fig, ax = plt.subplots(figsize=(14, 6))
ax.set_title(f"Rank distribution per database. Total: {numsrc}")
stats = scores_ann.groupby("database", observed=True).agg({"rank": ["median", "count"]})
stats.columns = stats.columns.droplevel(0)
stats = stats.sort_values("median")
sns.boxplot(
    data=scores_ann,
    y="rank",
    x="database",
    order=stats.index,
    ax=ax,
    width=1.0,
    gap=0.1,
)

for i, row in enumerate(stats.itertuples()):
    ax.text(
        i,
        row.median,
        f"n={row.count}",
        size="x-small",
        weight="semibold",
        horizontalalignment="center",
        verticalalignment="bottom",
    )
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(analysis_path / "db_rank.pdf", bbox_inches="tight")
plt.close("all")

recall = (
    scores.query("y_true")
    .sort_values(["genes", "rank"])
    .drop_duplicates(["genes"])["rank"]
    .value_counts()
    .sort_index()
    .cumsum()
)
recall.to_csv(analysis_path / "recall.csv")

rnds = []
counts = scores.groupby("genes")["y_true"].sum()
for _ in range(100):
    vals = {}
    for k in counts.values:
        if k not in vals:
            vals[k] = np.any(
                (np.random.choice(numsrc, [10000, k]) < 15), axis=-1
            ).mean()
    rnds.append(counts.map(vals).mean())
print("Random baseline")
print(pd.Series(rnds).describe())
print(all_metrics)
