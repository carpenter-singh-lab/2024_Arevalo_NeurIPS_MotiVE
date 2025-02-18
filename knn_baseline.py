import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from evaluate import add_rank

# results_path = "optimize/orf/target/bipartite/gnn/e1d65a/cartesian/test/results.parquet"
results_path = "outputs/orf/target/bipartite/gnn/8155db/cartesian/test/results.parquet"
edges_path = "data/bipartite/orf/target/s_t_labels.parquet"
tgt_path = "data/bipartite/orf/target.parquet"

comparison = []
for k in range(1, 11):
    edges = pd.read_parquet(edges_path)
    tgt = pd.read_parquet(tgt_path)
    train = edges.query("subset == 'train'")
    train = train.pivot(index="target", columns="source", values="subset").notna()
    X_train = tgt.iloc[train.index].values
    y_train = train.values
    clf = KNeighborsClassifier(metric="cosine", n_neighbors=k)
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
    gnn["y_pred"] = gnn["rank"] <= k
    gnn = gnn[gnn.source.isin(knn.source) & gnn.target.isin(knn.target)]
    gnn_score = classification_report(gnn["y_true"], gnn["y_pred"], output_dict=True)[
        "True"
    ]
    knn_score = classification_report(knn["y_true"], knn["y_pred"], output_dict=True)[
        "True"
    ]
    gnn_score.update(
        {
            "success@k": gnn.query("y_true and y_pred")["target"].nunique(),
            "Model": "gnn",
            "K": k,
        }
    )
    knn_score.update(
        {
            "success@k": knn.query("y_true and y_pred")["target"].nunique(),
            "Model": "knn",
            "K": k,
        }
    )
    comparison.append(gnn_score)
    comparison.append(knn_score)
comparison = pd.DataFrame(comparison).drop(columns="support")
comparison = comparison.melt(
    id_vars=["Model", "K"], var_name="metric", value_name="Score"
)
comparison["metric"] = comparison["metric"].str.title()

plt.close("all")
with plt.rc_context({"font.size": 17}):  # Set font size within this context
    sns.relplot(
        comparison,
        x="K",
        y="Score",
        hue="Model",
        col="metric",
        kind="line",
        marker="o",
        col_wrap=2,
        facet_kws={"sharey": False, "sharex": True},
    ).set_titles("{col_name}")
    plt.savefig("knn_comparison.pdf", bbox_inches="tight")
