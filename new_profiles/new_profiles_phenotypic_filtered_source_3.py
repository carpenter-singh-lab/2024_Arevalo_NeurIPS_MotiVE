from collections.abc import Iterable

import copairs.map as copairs
import numpy as np
import pandas as pd

POSCON_CODES = [
    "JCP2022_012818",
    "JCP2022_050797",
    "JCP2022_064022",
    "JCP2022_035095",
    "JCP2022_046054",
    "JCP2022_025848",
    "JCP2022_037716",
    "JCP2022_085227",
    "JCP2022_805264",
    "JCP2022_915132",
]
NEGCON_CODES = [
    "JCP2022_800001",
    "JCP2022_800002",
    "JCP2022_033924",
    "JCP2022_915131",
    "JCP2022_915130",
    "JCP2022_915129",
    "JCP2022_915128",
]


def add_pert_type(meta: pd.DataFrame, col: str = "Metadata_pert_type"):
    meta[col] = "trt"
    meta.loc[meta["Metadata_JCP2022"].isin(POSCON_CODES), col] = "poscon"
    meta.loc[meta["Metadata_JCP2022"].isin(NEGCON_CODES), col] = "negcon"
    meta[col] = meta[col].astype("category")


def find_feat_cols(cols: Iterable[str]):
    """Find column names for features"""
    feat_cols = [c for c in cols if not c.startswith("Meta")]
    return feat_cols


def find_meta_cols(cols: Iterable[str]):
    """Find column names for metadata"""
    meta_cols = [c for c in cols if c.startswith("Meta")]
    return meta_cols


def split_parquet(
    dframe_path, features=None
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    dframe = pd.read_parquet(dframe_path)
    if features is None:
        features = find_feat_cols(dframe)
    vals = np.empty((len(dframe), len(features)), dtype=np.float32)
    for i, c in enumerate(features):
        vals[:, i] = dframe[c]
    meta = dframe[find_meta_cols(dframe)].copy()
    return meta, vals, features


def _index(meta, ignore_codes=None, include_codes=None):
    """Select samples to be used in mAP computation"""
    index = meta["Metadata_pert_type"] != "poscon"
    valid_cmpd = meta.loc[index, "Metadata_JCP2022"].value_counts()
    valid_cmpd = valid_cmpd[valid_cmpd.between(2, 1000)].index
    if include_codes:
        valid_cmpd = valid_cmpd.union(include_codes)
    index &= meta["Metadata_JCP2022"].isin(valid_cmpd)
    # TODO: This compound has many more replicates than any other. ignoring it
    # for now. This filter should be done early on.
    index &= meta["Metadata_JCP2022"] != "JCP2022_033954"
    if ignore_codes:
        index &= ~meta["Metadata_JCP2022"].isin(ignore_codes)
    return index.values


def _group_negcons(meta: pd.DataFrame):
    """
    Hack to avoid mAP computation for negcons. Assign a unique id for every
    negcon so that no pairs are found for such samples.
    """
    negcon_ix = meta["Metadata_JCP2022"].isin(NEGCON_CODES)
    n_negcon = negcon_ix.sum()
    negcon_ids = [f"negcon_{i}" for i in range(n_negcon)]
    pert_id = meta["Metadata_JCP2022"].astype("category").cat.add_categories(negcon_ids)
    pert_id[negcon_ix] = negcon_ids
    meta["Metadata_JCP2022"] = pert_id


def average_precision_negcon(parquet_path, ap_path):
    meta, vals, _ = split_parquet(parquet_path)
    add_pert_type(meta)
    ix = _index(meta, include_codes=NEGCON_CODES)
    meta = meta[ix].copy()
    vals = vals[ix]
    _group_negcons(meta)
    result = copairs.average_precision(
        meta,
        vals,
        pos_sameby=["Metadata_JCP2022"],
        # pos_diffby=['Metadata_Well'],
        pos_diffby=[],
        neg_sameby=["Metadata_Plate"],
        neg_diffby=["Metadata_pert_type", "Metadata_JCP2022"],
        batch_size=20000,
    )
    result = result.query('Metadata_pert_type!="negcon"')
    result.reset_index(drop=True).to_parquet(ap_path)


def mean_average_precision(ap_path, map_path, threshold=0.05):
    ap_scores = pd.read_parquet(ap_path)

    map_scores = copairs.mean_average_precision(
        ap_scores, "Metadata_JCP2022", threshold=threshold, null_size=10000, seed=0
    )
    map_scores.to_parquet(map_path)


meta = pd.read_csv("plate.csv.gz")
df = pd.read_parquet("profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet")
df["Metadata_PlateType"] = df["Metadata_Plate"].map(
    meta.set_index("Metadata_Plate")["Metadata_PlateType"]
)
df = df.query("Metadata_PlateType.isin(['ORF', 'CRISPR']) or Metadata_Source=='source_3'")
df.to_parquet("prof_source_3.parquet")

average_precision_negcon("prof_source_3.parquet", "ap_scores.parquet")
mean_average_precision("ap_scores.parquet", "map_scores.parquet", 0.1)
mscores = pd.read_parquet("map_scores.parquet")

df = pd.read_parquet("prof_source_3.parquet")
below_p = set(mscores.query("below_p")["Metadata_JCP2022"])
df = df.query("Metadata_JCP2022 in @below_p")
df["Metadata_PlateType"] = df["Metadata_Plate"].map(
    meta.set_index("Metadata_Plate")["Metadata_PlateType"]
)

df.query("Metadata_PlateType=='ORF'").to_parquet(
    "inputs/orf/features.parquet", index=False
)
df.query("Metadata_PlateType=='CRISPR'").to_parquet(
    "inputs/crispr/features.parquet", index=False
)
df.query(
    "Metadata_PlateType.isin(['COMPOUND', 'TARGET2'])"
).to_parquet("inputs/compound/features.parquet", index=False)

pd.read_parquet("inputs/compound/features.parquet")
