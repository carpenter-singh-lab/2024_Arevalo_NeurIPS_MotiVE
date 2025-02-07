import numpy as np
import pandas as pd
import statsmodels.distributions as sm
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph

tgt = pd.read_parquet("data/bipartite/orf/target.parquet")
tmap = pd.read_parquet("data/bipartite/orf/target_map.parquet")[0]
X = tgt.divide(np.linalg.norm(tgt, axis=1), axis=0)
A = kneighbors_graph(tgt, metric="cosine", n_neighbors=1).toarray()

row_indices, col_indices = np.triu_indices(A.shape[0])
# row_indices, col_indices = np.array(
#     list(itertools.product(range(len(tgt)), range(len(tgt))))
# ).T
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

preds_path = "outputs/orf/target/bipartite/gnn/6756bb/cartesian/test/results.parquet"
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
num_success = dframe.query(f"y_true and rank<={k}").target.nunique()
print(f"Total number of success genes: {num_success}")

# Questions: is the morph similarity of target_y and target_x greater than alpha?
data = []
for alpha in np.arange(0.1, 1.0, 0.1):
    per_dist = tgt_dist.query("gene_dist.abs() > @alpha")
    num_tgt_per_dist = conns.merge(
        per_dist, on=["target_x", "target_y"]
    ).target_x.nunique()
    print(
        f"Number of success genes that have a similar(alpha={alpha:.1f}) gene and such gene is connected to the compound: {num_tgt_per_dist}"
    )
    data.append({"x": alpha, "y": num_tgt_per_dist})
data = pd.DataFrame(data)
# Create the plot
plt.close("all")
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
plt.plot(data["x"], data["y"], marker="o")
plt.axhline(y=num_success, color="r", linestyle="--", label="GNN score")
# Set titles and labels
plt.xlabel(r"Similarity threshold $\alpha$")
plt.ylabel("Number of success genes (Robhan metric)")
plt.title(
    r"Predicted genes having a similar (>$\alpha$) gene that is connected to the compound"
)
# Add legend
plt.legend()
# Show plot
plt.grid(True)  # Optional: Add grid for better readability
plt.savefig("sims.pdf", bbox_inches="tight")


# Questions: does target_y is in the K-nn of target_x?
per_neigh = tgt_dist.merge(neigh, on=["target_x", "target_y"])
num_tgt_per_neigh = conns.merge(
    per_neigh, on=["target_x", "target_y"]
).target_x.nunique()
print(
    f"Number of sucess genes that have a similar(knn=1) gene and such gene is connected to the compound: {num_tgt_per_neigh}"
)

plt.close("all")
similarity_scores = tgt_dist["gene_dist"]
# Define multiple threshold values
thresholds = np.round(np.arange(-0.6, 1, 0.2), 1)
cmap = plt.get_cmap("tab10")  # Using tab10 colormap, which has 10 distinct colors
colors = [cmap(i) for i in range(len(thresholds))]  # Get colors from the colormap

num_genes = len(tgt)
fig, ax1 = plt.subplots()
hist_color = cmap(0)
n, bins, patches = ax1.hist(
    similarity_scores,
    bins=50,
    # density=True,
    alpha=0.6,
    color=hist_color,
    edgecolor="black",
)
ax1.set_ylabel("Frequency", color=hist_color, labelpad=10)
ax2 = ax1.twinx()
cdf_color = cmap(1)
ecdf = sm.ECDF(similarity_scores)
x_cdf = np.linspace(min(similarity_scores), max(similarity_scores), num=100)
y_cdf = ecdf(x_cdf)
ax2.plot(x_cdf, y_cdf, color=cdf_color, linewidth=2, label="CDF")
ax2.set_ylabel("Cumulative Probability (CDF)", color=cdf_color)
ax2.set_ylim([0, 1.05])
ax3 = ax1.secondary_xaxis("top")
ax3.tick_params(axis="x", labelsize="smaller")
cdf_tick_labels = [f"{100 - ecdf(t)*100:.2f}%" for t in thresholds]
ax3.set_xticks(thresholds)
ax3.set_xticklabels(cdf_tick_labels)
ax3.set_xlabel("1 - CDF", labelpad=5)
ax1.set_xlabel("Cosine Similarity")
plt.title(f"Gene-Gene Similarities of ORF genes (num_genes)", pad=40)
ax1.grid(axis="y", alpha=0.75)
plt.subplots_adjust(top=0.85)
plt.legend()  # Legend removed
plt.savefig("hist.pdf", bbox_inches="tight")
