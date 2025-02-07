from itertools import zip_longest

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from umap import UMAP

from mworkflow import DEVICE, init


@torch.inference_mode
def umap(config_path, model_path, umap_path):
    config, model, loaders = init(config_path)
    best_params = torch.load(model_path, weights_only=True)
    model.load_state_dict(best_params["model_state_dict"])
    model.eval()

    data = loaders["test"].loader.data
    src = data["source"].node_id.cpu().numpy()
    tgt = data["target"].node_id.cpu().numpy()
    edges = np.fromiter(zip_longest(src, tgt, fillvalue=0), dtype=(int, 2))
    edges = torch.tensor(edges).T.contiguous()
    data["binds"].edge_label_index = edges
    seen_src = data["binds"].edge_index[0].unique().cpu().numpy()
    seen_src = np.in1d(np.arange(len(data["source"].node_id)), seen_src)
    seen_tgt = data["binds"].edge_index[1].unique().cpu().numpy()
    seen_tgt = np.in1d(np.arange(len(data["target"].node_id)), seen_tgt)

    activations = {}

    def hook(model, input, output):
        nonlocal activations
        activations = output

    model.gnn.register_forward_hook(hook)
    model(data.to(DEVICE))
    source_embd = activations["source"].cpu().numpy()
    target_embd = activations["target"].cpu().numpy()

    embd = np.concatenate([source_embd, target_embd])
    embd = UMAP().fit_transform(embd)

    df = pd.DataFrame(embd, columns=["x", "y"])
    df["node_type"] = ["source"] * len(source_embd) + ["target"] * len(target_embd)
    df["node_id"] = df.index - (len(source_embd) * (df["node_type"] != "source"))
    df["seen"] = np.concatenate([seen_src, seen_tgt])

    df.to_parquet(umap_path, index=False)


def scatter(umap_path, scatter_path):
    df = pd.read_parquet(umap_path)
    pallete = {False: "blue", True: (0.5, 0.5, 0.5, 0.2)}
    sns.scatterplot(
        df.sort_values("seen", ascending=False),
        x="x",
        y="y",
        hue="seen",
        palette=pallete,
        style="node_type",
    )
    plt.savefig(scatter_path, bbox_inches="tight")
