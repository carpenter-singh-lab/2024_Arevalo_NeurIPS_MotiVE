import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

st = pd.read_parquet(
    "storage/data_assembled/st_expanded/orf/s_t_labels.parquet"
).astype(np.int32)
tt = pd.read_parquet(
    "storage/data_assembled/st_expanded/orf/t_t_labels.parquet"
).astype(np.int32)
ss = pd.read_parquet(
    "storage/data_assembled/st_expanded/orf/s_s_labels.parquet"
).astype(np.int32)

n = max(st["target"].max(), tt.max(None))
m = max(st["source"].max(), ss.max(None))

G = nx.Graph()
G.add_edges_from(tt.values.tolist())
G.add_edges_from((ss.values + n).tolist())
G.add_edges_from((st.values + [n, 0]).tolist())
connected_components = list(nx.connected_components(G))

largest_cc = max(connected_components, key=len)
G_largest = G.subgraph(largest_cc).copy()

n_clusters = 2
A = nx.adjacency_matrix(G_largest).astype(np.int32).toarray()
clustering = SpectralClustering(
    n_clusters=n_clusters,
    assign_labels="discretize",
    random_state=0,
    affinity="precomputed",
)
labels = clustering.fit_predict(A)
node_to_cluster = {node: int(label) for node, label in zip(G.nodes(), labels)}

ct = pd.Series(node_to_cluster).to_frame()
ct.columns = ["cluster_id"]
ct["node_type"] = ct.index >= n
ct.eval("node_id = @ct.index - @n * node_type", inplace=True)
ct["node_type"] = ct["node_type"].map({False: "target", True: "source"})

tmap = ct.query("node_type=='target'").set_index("node_id")["cluster_id"]
smap = ct.query("node_type=='source'").set_index("node_id")["cluster_id"]

lbls = ss.map(smap.get)
ss_ct = ss.join(lbls, rsuffix="_ct").dropna().copy()

lbls = tt.map(tmap.get)
tt_ct = tt.join(lbls, rsuffix="_ct").dropna().copy()

st_ct = st.copy()
st_ct["source_ct"] = st_ct["source"].map(smap)
st_ct["target_ct"] = st_ct["target"].map(tmap)
st_ct.dropna(inplace=True)

print(tt_ct.eval("target_a_ct + target_b_ct").value_counts())
print(ss_ct.eval("source_a_ct + source_b_ct").value_counts())
print(st_ct.eval("source_ct + target_ct").value_counts())
