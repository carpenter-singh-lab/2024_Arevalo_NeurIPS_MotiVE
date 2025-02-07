import pandas as pd
import PyComplexHeatmap as pch
import seaborn as sns
from matplotlib import pyplot as plt

from mworkflow import init


def cm_mlib(pivot):
    clustergrid = sns.clustermap(pivot, cmap="vlag")
    pivot = pivot.iloc[clustergrid.dendrogram_row.reordered_ind]
    columns_ord = pivot.columns[clustergrid.dendrogram_col.reordered_ind]
    pivot = pivot[columns_ord]
    return pivot


def heatmap(config_path, scores_path, plot_path):
    config, model, loaders = init(config_path)
    scores = pd.read_parquet(scores_path)
    if config["leave_out"] == "target":
        row_var, col_var = "source", "target"
        leave_out_ix = 1
    else:
        row_var, col_var = "target", "source"
        leave_out_ix = 0

    pivot = scores.pivot(index=row_var, columns=col_var, values="score")
    data = loaders["test"].loader.data
    seen = data["binds"].edge_index[1 - leave_out_ix].unique().cpu().numpy()
    seen = pd.Series(index=pivot.index, data=pivot.index.isin(seen))

    source_ha = pch.HeatmapAnnotation(
        axis=0,
        orientation="right",
        Seen=pch.anno_simple(seen.astype(str), legend=True),
    )

    pch.ClusterMapPlotter(
        data=pivot,
        col_cluster=True,
        row_cluster=True,
        row_dendrogram=True,
        col_dendrogram=True,
        right_annotation=source_ha,
        cmap="vlag",
    )
    plt.savefig(plot_path, bbox_inches="tight")
