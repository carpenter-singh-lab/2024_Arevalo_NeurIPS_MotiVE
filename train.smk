import json
from utils.utils import hashname
import mworkflow
import evaluate

config["hash"] = hashname(config)


include: "plot.smk"


wildcard_constraints:
    subset="train|valid|test",
    leave_out="source|target|random",
    node="source|target",
    target_type="orf|crispr",
    graph_type="bipartite|s_expanded|t_expanded|st_expanded",
    hash=r"\b[0-9a-f]{6}\b",


rule all:
    input:
        expand(
            "{output_path}/{target_type}/{leave_out}/{graph_type}/{model}/{hash}/{infer_mode}/test/metrics.parquet",
            **config,
            infer_mode=["sampled", "cartesian"],
        ),
        expand(
            "{output_path}/{target_type}/{leave_out}/{graph_type}/{model}/{hash}/cartesian/test/analysis/waterfall.pdf",
            **config,
        ),


rule metrics:
    input:
        expand(
            "{{output_path}}/{{infer_mode}}/{{subset}}/metrics/{metric}.npy",
            metric=[
                "acc",
                "roc_auc",
                "hits_at_500",
                "precision_at_500",
                "f1",
                "mrr",
                "bce",
                "map_source",
                "map_target",
                "success_at_0.01_source_num",
                "success_at_0.01_source_pct",
                "success_at_0.01_target_num",
                "success_at_0.01_target_pct",
            ],
        ),
        "{output_path}/config.json",
    output:
        "{output_path}/{infer_mode}/{subset}/metrics.parquet",
    run:
        evaluate.collate(*input, *output, infer_mode=wildcards.infer_mode)


rule save_config:
    output:
        expand(
            "{output_path}/{target_type}/{leave_out}/{graph_type}/{model}/{hash}/config.json",
            **config,
        ),
    run:
        with open(*output, "w") as f:
            json.dump(config, f, indent=4)


rule train:
    input:
        "{output_path}/config.json",
    output:
        "{output_path}/weights.pt",
    run:
        mworkflow.train(*input, *output)


rule infer_sampled:
    input:
        config_path="{output_path}/config.json",
        model_path="{output_path}/weights.pt",
    output:
        "{output_path}/sampled/{subset}/results.parquet",
    run:
        mworkflow.infer_sampled(
            input.config_path, input.model_path, wildcards.subset, *output
        )


rule infer_cartesian:
    input:
        config_path="{output_path}/config.json",
        model_path="{output_path}/weights.pt",
    output:
        "{output_path}/cartesian/{subset}/results.parquet",
    run:
        mworkflow.infer_cartesian(
            input.config_path, input.model_path, wildcards.subset, *output
        )


rule accuracy:
    input:
        "{output_path}/{subset}/results.parquet",
    output:
        "{output_path}/{subset}/metrics/acc.npy",
    run:
        evaluate.accuracy(*input, *output)


rule roc_auc:
    input:
        "{output_path}/{subset}/results.parquet",
    output:
        "{output_path}/{subset}/metrics/roc_auc.npy",
    run:
        evaluate.accuracy(*input, *output)


rule hits_at_k:
    input:
        "{output_path}/{subset}/results.parquet",
    output:
        "{output_path}/{subset}/metrics/hits_at_{k}.npy",
    run:
        evaluate.hits_at_k(*input, int(wildcards.k), *output)


rule precision_at_k:
    input:
        "{output_path}/{subset}/results.parquet",
    output:
        "{output_path}/{subset}/metrics/precision_at_{k}.npy",
    run:
        evaluate.precision_at_k(*input, int(wildcards.k), *output)


rule f1:
    input:
        "{output_path}/{subset}/results.parquet",
    output:
        "{output_path}/{subset}/metrics/f1.npy",
    run:
        evaluate.f1(*input, *output)


rule mrr:
    input:
        "{output_path}/{subset}/results.parquet",
    output:
        "{output_path}/{subset}/metrics/mrr.npy",
    run:
        evaluate.mrr(*input, *output)


rule bce:
    input:
        "{output_path}/{subset}/results.parquet",
    output:
        "{output_path}/{subset}/metrics/bce.npy",
    run:
        evaluate.bce(*input, *output)


rule average_precision:
    input:
        "{output_path}/{subset}/results.parquet",
    output:
        "{output_path}/{subset}/metrics/average_precision.parquet",
    run:
        evaluate.average_precision(*input, *output)


rule mean_average_precision:
    input:
        "{output_path}/{subset}/metrics/average_precision.parquet",
    output:
        "{output_path}/{subset}/metrics/map_{node}.npy",
    run:
        evaluate.mean_average_precision(*input, wildcards.node, *output)


rule success_at_k_ratio:
    input:
        "{output_path}/{subset}/results.parquet",
    output:
        "{output_path}/{subset}/metrics/success_at_{k}_{node}_num.npy",
        "{output_path}/{subset}/metrics/success_at_{k}_{node}_pct.npy",
    run:
        evaluate.success_at_k_ratio(*input, wildcards.node, float(wildcards.k), *output)
