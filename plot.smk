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


rule umap:
    input:
        "{output_path}/config.json",
        "{output_path}/weights.pt",
    output:
        "{output_path}/umap.parquet",
    run:
        plot.umap(*input, *output)


rule scatter:
    input:
        "{output_path}/umap.parquet",
    output:
        "{output_path}/scatter.png",
    run:
        plot.scatter(*input, *output)


rule bipartite_target_knn_baseline:
    input:
        "{output_path}/{infer_mode}/{subset}/results.parquet",
    output:
        "{output_path}/{infer_mode}/{subset}/analysis/bipartite_target_knn_baseline.pdf",
    run:
        plot.bipartite_target_knn_baseline(*input, *output)
