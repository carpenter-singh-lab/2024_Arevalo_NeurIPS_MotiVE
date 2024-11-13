"""
Helper functions
"""

import hashlib
import json
from pathlib import Path


def hashname(config: dict):
    """Create hashname from a configuration id"""
    utf8_encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    data_md5 = hashlib.md5(utf8_encoded).hexdigest()[:6]
    return data_md5


class PathLocator:
    """Define output locations given a configuration"""

    def __init__(self, config_path: str | Path, output_path: str | Path):
        with open(config_path, encoding="utf8") as freader:
            self.config = json.load(freader)
            self.hashid = hashname(self.config)
            graph_type = self.config["graph_type"]
            split = self.config["data_split"]
            tgt_type = self.config["target_type"]
            model = self.config["model_name"]

        root_dir = (
            Path(output_path) / tgt_type / split / graph_type / model / self.hashid
        )
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        self.config_path = root_dir / "config.json"
        if not self.config_path.exists():
            with self.config_path.open("w", encoding="utf8") as f_out:
                json.dump(self.config, f_out)

        # save model and training summary
        self.summary_path = root_dir
        self.model_path = root_dir / "weights.pth"
        self.valid_metrics_path = root_dir / "validation_metrics.csv"
        self.test_metrics_path = root_dir / "test_metrics.csv"
        self.test_results_path = root_dir / "test_results.parquet"
        self.best_threshold = 0.0

    def __str__(self):
        attrs = vars(self)
        string = ", ".join("%s: %s" % item for item in attrs.items())
        return string
