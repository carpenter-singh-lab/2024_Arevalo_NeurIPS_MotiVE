import itertools
import json

import pandas as pd
from matplotlib import pyplot as plt
from PyComplexHeatmap import (
    HeatmapAnnotation,
    anno_barplot,
    anno_label,
    oncoPrintPlotter,
)

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


def waterfall(config_path, preds_path, waterfall_path):
    with open(config_path) as f:
        config = json.load(f)
    leave_out = config["leave_out"]
    graph_type = config["graph_type"]
    target_type = config["target_type"]

    scores = pd.read_parquet(preds_path)
    scores["rank"] = (
        scores.groupby(leave_out)["score"]
        .rank(method="min", ascending=False, pct=False)
        .astype(int)
    )

    smap = pd.read_parquet(f"data/{graph_type}/{target_type}/source_map.parquet")
    tmap = pd.read_parquet(f"data/{graph_type}/{target_type}/target_map.parquet")
    scores["compound"] = smap.index[scores["source"]]
    scores["gene"] = tmap.index[scores["target"]]

    ann = pd.read_parquet("inputs/annotations/compound_gene.parquet")
    ann["compound"] = ann["inchikey"].fillna("").str[:14]
    ann.rename(columns={"target": "gene"}, inplace=True)

    scores_ann = scores.query("y_true").merge(ann, on=["compound", "gene"], how="left")
    scores_ann["rel_type"] = scores_ann["rel_type"].apply(lambda x: mapper.get(x, x))
    scores_ann.head()

    plt.figure(figsize=(12, 8))
    wf = (
        scores_ann.query("rank<=15")[["gene", "compound", "rel_type"]]
        .drop_duplicates()
        .copy()
    )
    wf["value"] = 1
    counts = wf["rel_type"].value_counts()
    cart = itertools.product(wf["gene"].unique(), wf["compound"].unique())
    cart = pd.DataFrame(cart, columns=["gene", "compound"])
    cart["value"] = 0
    cart.set_index(["gene", "compound"], inplace=True)
    known = wf[["gene", "compound"]].drop_duplicates().itertuples(index=False)
    cart.loc[known, "value"] = 1
    cart = cart.query("value==0").reset_index().copy()
    cart["rel_type"] = wf["rel_type"].iloc[0]
    wf = pd.concat([wf, cart])
    wf = wf.sort_values(by="rel_type", key=lambda x: x.map(counts), ascending=False)

    wf = (
        wf.pivot(index=["gene", "compound"], columns="rel_type", values="value")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    cols = list(counts.index)
    wf = wf[["gene", "compound"] + cols]
    print(wf.head())

    colors = plt.cm.tab20(range(20))
    colors = [
        f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for c in colors
    ]
    colors = colors[: len(cols)]

    plt.close("all")
    plt.figure(figsize=(18, 14))
    gene_counts = wf.groupby("gene")[cols].sum()
    inchi_counts = wf.groupby("compound")[cols].sum()
    inchi_counts_all = (
        scores.query("rank<=15")["compound"].value_counts()[inchi_counts.index]
        / scores["gene"].nunique()
    )
    inchi_counts_all = inchi_counts_all.apply("{:,.2%}".format)
    top_annotation = HeatmapAnnotation(
        axis=1, rel_type=anno_barplot(gene_counts, colors=colors), legend=False
    )
    right_annotation = HeatmapAnnotation(
        axis=0,
        orientation="right",
        compound=anno_barplot(inchi_counts, colors=colors, legend=False),
        hits=anno_label(inchi_counts_all, colors="black"),
        label_kws={"visible": False},
    )
    oncoPrintPlotter(
        data=wf,
        y="compound",
        x="gene",
        values=cols,
        colors=colors,
        label="rel_type",
        show_rownames=True,
        show_colnames=True,
        row_names_side="left",
        top_annotation=top_annotation,
        right_annotation=right_annotation,
        legend_width=100,
    )
    plt.savefig(waterfall_path, bbox_inches="tight")
