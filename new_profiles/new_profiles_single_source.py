import pandas as pd

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
df.query(
    "Metadata_PlateType.isin(['COMPOUND', 'TARGET2']) and Metadata_Source=='source_3'"
).to_parquet("inputs/compound/features.parquet", index=False)

pd.read_parquet("inputs/compound/features.parquet")
