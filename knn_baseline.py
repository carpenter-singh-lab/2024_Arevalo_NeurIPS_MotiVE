import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from evaluate import add_rank

# results_path = (
# "outputs/orf/target/st_expanded/gnn/bdf511/cartesian/test/results.parquet"
# )
results_path = "optimize/orf/target/bipartite/gnn/a85b58/cartesian/test/results.parquet"
edges_path = "data/bipartite/orf/target/s_t_labels.parquet"
tgt_path = "data/bipartite/orf/target.parquet"

edges = pd.read_parquet(edges_path)
tgt = pd.read_parquet(tgt_path)
train = edges.query("subset == 'train'")
train = train.pivot(index="target", columns="source", values="subset").notna()
X_train = tgt.iloc[train.index].values
y_train = train.values
clf = KNeighborsClassifier(metric="cosine", n_neighbors=10)
clf.fit(X_train, y_train)

test_tgt = edges.query("subset == 'test'")["target"].drop_duplicates()
X_test = tgt.iloc[test_tgt].values
y_prob = np.stack(clf.predict_proba(X_test))[:, :, 1].T
score = pd.DataFrame(y_prob, index=test_tgt, columns=train.columns)

# Keep only compounds in test. Fill samples with 0
# test_src = edges.query("subset=='test'")["source"].drop_duplicates()
# others = [c for c in test_src if c not in score]
# others = pd.DataFrame(index=score.index, columns=others, data=0.0)
# score = pd.concat([score, others], axis=1)
# score = score[test_src]

# Keep only compounds that are in train and test
test_src = edges.query("subset=='test'")["source"].drop_duplicates()
train_test_src = [c for c in test_src if c in score]
score = score[train_test_src]

score = score.melt(
    ignore_index=False, value_name="score", var_name="source"
).reset_index()
score = score.merge(edges, on=["source", "target"], how="left")
score["y_true"] = score["subset"].notna()
score["y_pred"] = score["score"] > 0
knn = score

gnn = pd.read_parquet(results_path)
add_rank(gnn, "target")
gnn["y_pred"] = gnn["rank"] <= 15
gnn = gnn[gnn.source.isin(knn.source) & gnn.target.isin(knn.target)]
print(classification_report(gnn["y_true"], gnn["y_pred"]))
print(classification_report(knn["y_true"], knn["y_pred"]))
print(confusion_matrix(gnn["y_true"], gnn["y_pred"]))
print(confusion_matrix(knn["y_true"], knn["y_pred"]))
