"""
wget ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz
wget ftp://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz
wget https://storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/tsv/hgnc_complete_set_2024-12-03.txt
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def unique_ids(G):
    mapper = {}
    for component_id, nodes in enumerate(nx.connected_components(G)):
        for node in nodes:
            mapper[node] = component_id

    return mapper


def get_synonyms():
    gene_ids = pd.read_csv("Homo_sapiens.gene_info.gz", sep="\t")
    synonyms = gene_ids[["Symbol", "Synonyms"]].drop_duplicates()
    synonyms = synonyms.query('Synonyms!="-"').set_index("Symbol")["Synonyms"]
    synonyms = synonyms.str.split("|").explode().reset_index()
    synonyms = synonyms.drop_duplicates("Synonyms", keep=False)
    return synonyms


fp = Path("gene2go_homo_sapiens.parquet")
if not fp.is_file():
    df = pd.read_csv("gene2go.gz", sep="\t")
    df[df["#tax_id"] == 9606].reset_index(drop=True).to_parquet(fp)
    # 9606 => homo sapiens
hgnc = pd.read_csv("hgnc_complete_set_2024-12-03.txt", sep="\t", low_memory=False)
info = pd.read_csv("Homo_sapiens.gene_info.gz", sep="\t")
syns = get_synonyms()

nodes = (
    set(hgnc["symbol"])
    | set(hgnc["entrez_id"].dropna().astype(int))
    | set(info["Symbol"])
    | set(info["GeneID"].dropna().astype(int))
    | set(syns["Symbol"])
    | set(syns["Synonyms"])
)

G = nx.Graph()
G.add_nodes_from(nodes)
for p, q in hgnc[["symbol", "entrez_id"]].dropna().itertuples(index=False):
    G.add_edge(p, int(q))
for p, q in info[["Symbol", "GeneID"]].dropna().itertuples(index=False):
    G.add_edge(p, int(q))
for p, q in syns[["Symbol", "Synonyms"]].dropna().itertuples(index=False):
    G.add_edge(p, q)

component_ids = unique_ids(G)
edges = pd.read_parquet("data/st_expanded/orf/target/s_t_labels.parquet")
tmap = pd.read_parquet("data/st_expanded/orf/target_map.parquet")
smap = pd.read_parquet("data/st_expanded/orf/target_map.parquet")
edges["inchikey"] = smap.index[edges["source"]]
edges["symbol"] = tmap.index[edges["target"]]

edges = edges.query("symbol in @component_ids")
edges = edges[["symbol", "subset"]].copy()
edges["component_id"] = edges["symbol"].map(component_ids)
slim = pd.read_parquet("gene2go_homo_sapiens.parquet")
slim["component_id"] = slim["GeneID"].map(component_ids)
df = slim[["GO_ID", "component_id"]].merge(edges, on="component_id")
df = df[["GO_ID", "component_id", "subset"]].drop_duplicates()

mloc = info.set_index("GeneID")["map_location"][lambda x: x != "-"].reset_index()
mloc["component_id"] = mloc["GeneID"].map(component_ids)
mloc = mloc.groupby(["component_id", "map_location"]).size().reset_index()

mloc[mloc.duplicated(subset=["component_id"], keep=False)]

# TODO: Add gene_info.map_location as another annotation

r"""
Symbol -> GID
GID -> GO (annotation)

For a given gene that got good results.
did it share annotations with another train gene?
in other words, are they connected by the same GO term?

GO:term -> nodes

Given two disjoint set of genes train, and test; a list of annotations (gene_a,
gene_b, rel_type); and a list of scores for the test set.

which of the genes in test set that pass a threshold k have any annotation.

is there a tuple (a, b, r) such that a \in train, b \in test ?
plot stats on r
"""
