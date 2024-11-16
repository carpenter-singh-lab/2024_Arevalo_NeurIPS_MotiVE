import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from model import create_model
from motive import get_loaders
from train import run_test, train_loop
from utils.evaluate import save_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init(config_path):
    with open(config_path) as f:
        config = json.load(f)
    train_loader, val_loader, test_loader = get_loaders(
        config["leave_out"], config["target_type"], config["graph_type"]
    )
    train_data = train_loader.loader.data
    model = create_model(config, train_data).to(DEVICE)
    loaders = {"train": train_loader, "valid": val_loader, "test": test_loader}
    return config, model, loaders


def train(config_path, model_path):
    config, model, loaders = init(config_path)
    train_loop(model, model_path, config, loaders["train"], loaders["valid"])


def infer_sampled(config_path, model_path, subset, preds_path):
    config, model, loaders = init(config_path)
    best_params = torch.load(model_path, weights_only=True)
    best_th = best_params["best_th"]
    model.load_state_dict(best_params["model_state_dict"])
    preds = run_test(model, loaders[subset], best_th)
    preds.to_parquet(preds_path)


def infer_cartesian(config_path, model_path, subset, preds_path):
    config, model, loaders = init(config_path)
    best_params = torch.load(model_path, weights_only=True)
    best_th = best_params["best_th"]
    model.load_state_dict(best_params["model_state_dict"])

    data = loaders[subset].loader.data
    src = data["binds"].edge_label_index[0].unique()
    tgt = data["binds"].edge_label_index[1].unique()
    edges = torch.cartesian_prod(src, tgt).T.contiguous()
    gt_edges = data["binds"].edge_label_index
    y_true = torch.any(torch.all(edges.T[:, None] == gt_edges.T, axis=-1), axis=-1)
    data["binds"].edge_label_index = edges
    data["binds"].edge_label = y_true
    with torch.inference_mode():
        model.eval()
        logits = model(data.to(DEVICE))
        scores = torch.sigmoid(logits)

        edges = edges.cpu().numpy()
        y_true = y_true.cpu().numpy()
        logits = logits.cpu().numpy()
        scores = scores.cpu().numpy()
    scores = pd.DataFrame(
        {
            "source": edges[0],
            "target": edges[1],
            "score": scores,
            "logits": logits,
            "y_true": y_true,
            "y_pred": logits > best_th,
        }
    )
    scores.to_parquet(preds_path)


def register_tensorboard(config_path, scores_path):
    # Register results in tensorboards
    summary_path = Path(model_path).parent
    writer = SummaryWriter(log_dir=summary_path, comment=config["model"])
    writer.add_hparams(
        config,
        {f"test/{k}": v for k, v in test_scores.items()},
        run_name="./",
    )
    save_metrics(test_scores, test_metrics_path)
