import plot


rule waterfall:
    input:
        "{output_path}/config.json",
        "{output_path}/{infer_mode}/{subset}/results.parquet",
    output:
        "{output_path}/{infer_mode}/{subset}/analysis/waterfall.pdf",
    run:
        plot.waterfall(*input, *output)


rule heatmap:
    input:
        "{output_path}/config.json",
        "{output_path}/{infer_mode}/{subset}/results.parquet",
    output:
        "{output_path}/{infer_mode}/{subset}/analysis/heatmap.png",
    run:
        plot.heatmap(*input, *output)
