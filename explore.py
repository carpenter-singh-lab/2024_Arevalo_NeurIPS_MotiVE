import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter

from plot import contour
from train import SEED
from utils.utils import hashname


def generate_params(num_opts: int, params_path: str):
    config_search = []
    rng = np.random.default_rng(SEED)
    for i in range(num_opts):
        hidden_channels = rng.choice([128, 256, 512, 1024])
        learning_rate = 10.0 ** rng.uniform(-6, -1)
        weight_decay = 10.0 ** rng.uniform(-5, 0)

        config_search.append((hidden_channels, learning_rate, weight_decay))

    params = pd.DataFrame(
        config_search, columns=["hidden_channels", "learning_rate", "weight_decay"]
    )

    params.to_csv(params_path, index=False)


def generate_configs(params_path: str, config):
    params = pd.read_csv(params_path)
    output_dir = Path(
        "{output_path}/{target_type}/{leave_out}/{graph_type}/{model}/".format(**config)
    )
    for i, param in params.iterrows():
        config.update(param)
        config["hash"] = hashname(config)
        config_path = output_dir / config["hash"] / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as fw:
            json.dump(config, fw, indent=4)


def contours(paths: [str], config):
    results = pd.concat(map(pd.read_parquet, paths))
    metrics = [c for c in results if c not in config]
    metrics.remove("infer_mode")
    writer = SummaryWriter(log_dir=config["output_path"])
    for criteria in metrics:
        writer.add_image(
            f"{criteria} exploration",
            contour(results, criteria),
            dataformats="HWC",
        )
    writer.flush()
