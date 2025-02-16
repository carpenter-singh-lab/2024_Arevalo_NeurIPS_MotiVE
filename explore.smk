include: "train.smk"


import explore


def metrics_input(wildcards):
    path = Path(config["output_path"])
    confs = [x.parent for x in path.rglob("config.json")]
    return (conf / "cartesian/valid/metrics.done" for conf in confs)


def contour_input(wildcards):
    path = Path(config["output_path"])
    confs = [x.parent for x in path.rglob("config.json")]
    return (conf / "cartesian/valid/metrics.parquet" for conf in confs)


rule contours_tensorboard:
    input:
        "{output_path}/exploration.done".format(**config),
        contour_input,
    run:
        explore.contours(input[1:], config)


checkpoint results:
    input:
        "{output_path}/config.done".format(**config),
        metrics_input,
    output:
        touch("{output_path}/exploration.done"),


checkpoint generate_configs:
    input:
        "{output_path}/param_search.parquet".format(**config),
    output:
        touch("{output_path}/config.done".format(**config)),
    run:
        explore.generate_configs(*input, config)


rule generate_params:
    output:
        "{output_path}/param_search.parquet".format(**config),
    run:
        explore.generate_params(config["num_search"], *output)
