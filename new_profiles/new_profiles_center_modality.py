"""
wget https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles/jump-profiling-recipe_2024_0224e0f/ALL/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet
wget https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz
wget https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/microscope_config.csv
"""
####################
# WARNING: Run this only once!
####################

from itertools import product

import pandas as pd

from motive.jump import map_broad_to_inchi, map_broad_to_symbol, select_features


def update_all_source_with_centered_features():
    sources = pd.read_parquet(
        "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet"
    )
    sources = map_broad_to_inchi(sources.copy(), "inputs/compound/meta.csv.gz")
    metacols = [c for c in sources.columns if c.startswith("Meta")]
    featcols = [c for c in sources.columns if not c.startswith("Meta")]

    # Centered_features
    sources[featcols] -= sources[featcols].mean()

    agg_funcs = {c: "first" for c in metacols}
    agg_funcs.update({c: "median" for c in featcols})
    profiles = sources.groupby("Metadata_JCP2022", observed=True).agg(agg_funcs)
    profiles.drop_duplicates("Metadata_InChIKey", inplace=True)
    profiles = select_features(profiles, "Metadata_InChIKey")
    profiles.to_parquet("data/all_source.parquet")


def update_all_target_with_centered_features(target_type: str):
    feats = pd.read_parquet(
        "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet"
    )
    feats = map_broad_to_symbol(feats, f"inputs/{target_type}/meta.csv.gz")
    metacols = [c for c in feats.columns if c.startswith("Meta")]
    featcols = [c for c in feats.columns if not c.startswith("Meta")]

    # Centered_features
    feats[featcols] -= feats[featcols].mean()

    agg_funcs = {c: "first" for c in metacols}
    agg_funcs.update({c: "median" for c in featcols})
    profiles = feats.groupby("Metadata_Symbol", observed=True).agg(agg_funcs)
    profiles = select_features(profiles, "Metadata_Symbol")
    profiles.to_parquet(f"data/{target_type}_all_target.parquet")


update_all_source_with_centered_features()
update_all_target_with_centered_features("orf")
update_all_target_with_centered_features("crispr")

graph_types = "bipartite st_expanded t_expanded s_expanded".split()
target_types = ["orf", "crispr"]
node_types = ["source", "target"]

for node, graph, tgt in product(node_types, graph_types, target_types):
    tgt_prefix = f"{tgt}_" if node != "source" else ""
    path = f"data/{graph}/{tgt}/{node}.parquet"
    master = pd.read_parquet(f"data/{tgt_prefix}all_{node}.parquet")
    subset = pd.read_parquet(path)
    subset2 = master.loc[subset.index].copy()
    assert subset2.index.symmetric_difference(subset.index).empty
    print(f"changing {path} from", subset.shape, "to", subset2.shape)
    subset2.to_parquet(path)
