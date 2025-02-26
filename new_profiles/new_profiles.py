"""
wget https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles/jump-profiling-recipe_2024_0224e0f/ALL/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet
wget https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz
wget https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/microscope_config.csv
"""

from itertools import product

import pandas as pd

from motive.jump import map_broad_to_inchi, select_features


def update_all_source_with_assembled_features():
    sources = pd.read_parquet(
        "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet"
    )
    sources = map_broad_to_inchi(sources.copy(), "inputs/compound/meta.csv.gz")
    metacols = [c for c in sources.columns if c.startswith("Meta")]
    featcols = [c for c in sources.columns if not c.startswith("Meta")]
    agg_funcs = {c: "first" for c in metacols}
    agg_funcs.update({c: "median" for c in featcols})
    profiles = sources.groupby("Metadata_JCP2022", observed=True).agg(agg_funcs)
    profiles.drop_duplicates("Metadata_InChIKey", inplace=True)
    profiles = select_features(profiles, "Metadata_InChIKey")
    profiles.to_parquet("data/all_source.parquet")


"""
df = pd.read_parquet(
    "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet"
)
meta = pd.read_csv("plate.csv.gz")
df["Metadata_PlateType"] = df["Metadata_Plate"].map(
    meta.set_index("Metadata_Plate")["Metadata_PlateType"]
)
df.query("Metadata_PlateType=='ORF'").to_parquet(
    "inputs/orf/features.parquet", index=False
)
df.query("Metadata_PlateType=='CRISPR'").to_parquet(
    "inputs/crispr/features.parquet", index=False
)
df.query("Metadata_PlateType.isin(['COMPOUND', 'TARGET2'])").to_parquet(
    "inputs/compound/features.parquet", index=False
)
update_all_source_with_assembled_features()
"""

####################
# WARNING: Run this only when all_source, orf_all_target crispr_all_target are
# updated.
####################
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
    print(f"changing {path} from ", subset.shape, " to ", subset2.shape)
    subset2.to_parquet(path)
