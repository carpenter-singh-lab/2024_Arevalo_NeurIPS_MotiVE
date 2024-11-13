import argparse
import os.path

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from model import create_model
from motive import get_loaders
from train import run_test, train_loop
from utils.evaluate import save_metrics
from utils.utils import PathLocator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def workflow(locator, eval_test=False):
    leave_out = locator.config["data_split"]
    tgt_type = locator.config["target_type"]
    graph_type = locator.config["graph_type"]
    num_epochs = locator.config["num_epochs"]
    train_loader, val_loader, test_loader = get_loaders(leave_out, tgt_type, graph_type)
    train_data = train_loader.loader.data
    model = create_model(locator, train_data).to(DEVICE)
    best_th = train_loop(model, locator, train_loader, val_loader, num_epochs)
    if eval_test:
        results, test_scores = run_test(model, test_loader, best_th)
        save_metrics(test_scores, locator.test_metrics_path)
        results.to_parquet(locator.test_results_path)
        writer = SummaryWriter(
            log_dir=locator.summary_path, comment=locator.config["model_name"]
        )
        writer.add_hparams(
            locator.config,
            {f"test/{k}": v for k, v in test_scores.items()},
            run_name="./",
        )
        print(test_scores)


def main():
    """Parse input params"""
    parser = argparse.ArgumentParser(
        description=("Train GNN with this config file"),
    )
    parser.add_argument("config_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    locator = PathLocator(args.config_path, args.output_path)
    if os.path.isfile(locator.test_results_path):
        print(f"{locator.test_results_path} exists. Skipping...")
        return
    workflow(locator, eval_test=True)


if __name__ == "__main__":
    main()
