import itertools

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph

tgt = pd.read_parquet("data/bipartite/orf/target.parquet")
tmap = pd.read_parquet("data/bipartite/orf/target_map.parquet")[0]
X = tgt.divide(np.linalg.norm(tgt, axis=1), axis=0)
A = kneighbors_graph(tgt, metric="cosine", n_neighbors=1).toarray()

# row_indices, col_indices = np.triu_indices(A.shape[0])
row_indices, col_indices = np.array(
    list(itertools.product(range(len(tgt)), range(len(tgt))))
).T
tgt_dist = X @ X.T
tgt_dist = pd.DataFrame(
    {
        "symbol_a": X.index[row_indices],
        "symbol_b": X.index[col_indices],
        "gene_dist": tgt_dist.values[row_indices, col_indices],
    }
)
tgt_dist["target_x"] = tgt_dist["symbol_a"].map(tmap)
tgt_dist["target_y"] = tgt_dist["symbol_b"].map(tmap)
tgt_dist = tgt_dist.query("symbol_a != symbol_b").copy().reset_index(drop=True)
neigh = pd.DataFrame(dict(zip(["target_x", "target_y"], np.nonzero(A))))

preds_path = (
    "outputs/source_3/orf/target/bipartite/gnn/edd022/cartesian/test/results.parquet"
)
node = "target"
dframe = pd.read_parquet(preds_path)
k = 15
dframe["rank"] = dframe.groupby(node)["score"].rank(
    method="min", ascending=False, pct=False
)
dframe["rank_pct"] = dframe.groupby(node)["score"].rank(
    method="min", ascending=False, pct=True
)
num = dframe.query(f"rank <= {k} and y_true")[node].nunique()
pct = num / dframe[node].nunique()
preds = dframe.query(f"rank <= {k} and y_true")

st = pd.read_parquet("data/bipartite/orf/s_t_labels.parquet")

# S_x -- T_x -- S_y -- T_y
conns = (
    dframe.query("y_pred and y_true")
    .merge(st, on="target")
    .merge(st, left_on="source_y", right_on="source")
)
conns = conns.query("target_x != target_y")

# T_x -- S -- T_y
# conns = dframe.query("y_pred and y_true").merge(st, on="source")
conns = dframe.query(f"y_true and rank<={k}").merge(st, on="source")
conns = conns.query("target_x != target_y")
# conns has target_x, target_y.

# Questions: is the morph similarity of target_y and target_x greater than alpha?
alpha = 0.4
per_dist = tgt_dist.query("gene_dist.abs() > @alpha")
num_tgt_per_dist = conns.merge(per_dist, on=["target_x", "target_y"]).target_x.nunique()
print(
    f"Number of success genes that have a similar(alpha={alpha}) gene and such gene is connected to the compound: {num_tgt_per_dist}"
)
# Questions: does target_y is in the K-nn of target_x?
per_neigh = tgt_dist.merge(neigh, on=["target_x", "target_y"])
num_tgt_per_neigh = conns.merge(
    per_neigh, on=["target_x", "target_y"]
).target_x.nunique()
print(
    f"Number of sucess genes that have a similar(knn=1) gene and such gene is connected to the compound: {num_tgt_per_neigh}"
)
num_success =  dframe.query(f"y_true and rank<={k}").target.nunique()
print(f"Total number of success genes: {num_success}")

