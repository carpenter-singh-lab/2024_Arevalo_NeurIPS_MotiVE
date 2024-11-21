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
            "{output_path}/{target_type}/{leave_out}/{graph_type}/{model}/{hash}/{infer_mode}/test/metrics.done",
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
            ],
        ),
        expand(
            "{{output_path}}/{{infer_mode}}/{{subset}}/metrics/{metric}_{node}.npy",
            metric=[
                "map",
                "phenotypic_activity",
                "success_at_15_num",
                "success_at_15_pct",
                "random_success_at_15_num",
                "random_success_at_15_pct",
            ],
            node=["source", "target"],
        ),
        "{output_path}/config.json",
    output:
        "{output_path}/{infer_mode}/{subset}/metrics.parquet",
    run:
        evaluate.collate(*input, *output, infer_mode=wildcards.infer_mode)


rule register_tensorboard:
    input:
        "{output_path}/config.json",
        "{output_path}/{infer_mode}/{subset}/metrics.parquet",
    output:
        touch("{output_path}/{infer_mode}/{subset}/metrics.done"),
    run:
        evaluate.register_tensorboard(*input, wildcards.subset)


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
        "{output_path}/{subset}/metrics/mean_average_precision.parquet",
    params:
        null_size=10000,
        threshold=0.1,
        seed=0,
    run:
        evaluate.mean_average_precision(*input, *output, **params)


rule phenotypic_activity:
    input:
        "{output_path}/{subset}/metrics/mean_average_precision.parquet",
    output:
        "{output_path}/{subset}/metrics/phenotypic_activity_{node}.npy",
        "{output_path}/{subset}/metrics/map_{node}.npy",
    run:
        evaluate.phenotypic_activity(*input, wildcards.node, *output)


rule success_at_k:
    input:
        "{output_path}/{subset}/results.parquet",
    output:
        "{output_path}/{subset}/metrics/success_at_{k}_num_{node}.npy",
        "{output_path}/{subset}/metrics/success_at_{k}_pct_{node}.npy",
    run:
        evaluate.success_at_k(*input, wildcards.node, int(wildcards.k), *output)


rule random_success_at_k:
    input:
        "{output_path}/{subset}/results.parquet",
    output:
        "{output_path}/{subset}/metrics/random_success_at_{k}_num_{node}.npy",
        "{output_path}/{subset}/metrics/random_success_at_{k}_pct_{node}.npy",
    run:
        evaluate.random_success_at_k(*input, wildcards.node, int(wildcards.k), *output)
