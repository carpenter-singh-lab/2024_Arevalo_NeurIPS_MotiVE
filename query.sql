DROP VIEW IF EXISTS results;


CREATE VIEW results AS
SELECT *
FROM read_parquet('./**/*metrics.parquet', filename=TRUE);


SELECT split_part(filename, '/', 3) AS DATA,
 (success_at_15_num_target / success_at_15_pct_target)::INT AS total_gene,
 (success_at_15_num_source / success_at_15_pct_source)::INT AS total_cmpd,
 round(map_target, 3) AS map_gene,
 round(phenotypic_activity_target, 3) AS pa_gene,
 success_at_15_num_target AS gene_num,
 round(success_at_15_pct_target, 3) AS 'success@15',
 round(random_success_at_15_pct_target, 3) AS rnd_baseline,
FROM results
WHERE infer_mode = 'cartesian'
ORDER BY map_gene DESC;
