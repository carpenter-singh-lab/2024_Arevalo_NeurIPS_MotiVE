# %run predict_all.py
top15 = scores.query("rank<=15")
plt.figure(figsize=(11, 6))
ax = sns.barplot(top15.inchi.value_counts())
plt.xticks(rotation=45, ha="right")
ax.set_xlabel("Compound")
ax.set_ylabel("Number of genes where the compound is in top-15")
plt.savefig(analysis_path / "top15_bar.pdf", bbox_inches="tight")
plt.close("all")

filtered_scores = scores
top15 = scores.query("rank<=15")
print(top15["inchi"].nunique())
redlist = [""]
while len(redlist) > 0:
    redlist = top15["inchi"].value_counts()[lambda x: x > 800].index
    filtered_scores = filtered_scores.query("inchi not in @redlist").copy()
    filtered_scores["rank"] = filtered_scores.groupby("genes")["score"].rank(
        method="min", ascending=False, pct=False
    )
    filtered_scores["rank_pct"] = filtered_scores.groupby("genes")["score"].rank(
        method="min", ascending=False, pct=True
    )
    num = filtered_scores.query(f"rank_pct <= {0.01} and y_true==1")["genes"].nunique()
    top15 = filtered_scores.query("rank<=15")
    print(top15["inchi"].nunique(), num, num / filtered_scores["genes"].nunique())

plt.figure(figsize=(14, 6))
ax = sns.barplot(top15.inchi.value_counts())
plt.xticks(rotation=45, ha="right", fontsize=7)
ax.set_xlabel("Compound")
ax.set_ylabel("Number of genes where the compound is in top-15")

plt.savefig(analysis_path / "top15_bar_filtered.pdf", bbox_inches="tight")
plt.close("all")

pivot = filtered_scores.pivot(index="inchi", columns="genes", values="rank")
clustergrid = sns.clustermap(pivot, cmap="vlag")
pivot = pivot.iloc[clustergrid.dendrogram_row.reordered_ind]
columns_ord = pivot.columns[clustergrid.dendrogram_col.reordered_ind]
pivot = pivot[columns_ord]
pivot.to_csv(analysis_path / "pivot_filtered.csv")
plt.savefig(analysis_path / "clustermap_filtered.png", bbox_inches="tight")

