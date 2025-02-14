import json

import torch

from model import create_model
from motive import get_cartesian_loader, get_loaders
from train import run_test, train_loop

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init(config_path):
    with open(config_path) as f:
        config = json.load(f)
    train_loader, val_loader, test_loader = get_loaders(
        config["leave_out"],
        config["target_type"],
        config["graph_type"],
        config["neg_ratio"],
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
    model.load_state_dict(best_params["model_state_dict"])
    preds = run_test(model, loaders[subset])
    preds.to_parquet(preds_path)


def infer_cartesian(config_path, model_path, subset, preds_path):
    config, model, loaders = init(config_path)
    best_params = torch.load(model_path, weights_only=True)
    model.load_state_dict(best_params["model_state_dict"])

    data = loaders[subset].loader.data
    loader = get_cartesian_loader(data, mode="label")
    preds = run_test(model, loader)
    preds.to_parquet(preds_path)
