"""
sweetbits.features
Statistical and mathematical engines for metagenomic classification quality feature extraction.
"""

import polars as pl
import numpy as np
import click
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from scipy.stats import t
from joltax import JolTree
from sweetbits.taxmath import calc_clade_sum
from sweetbits.utils import UNCLASSIFIED_TID, FILTERED_TID, load_sample_id_list, check_write_permission
from sweetbits.metadata import validate_sweetbits_file, get_standard_metadata, save_companion_metadata

# Hardcoded Correlation Parameters
CORR_TOP_PERCENT_FILTER = 1.0  # Remove top 1% of samples by coverage for the filtered correlation
CORR_TOP_FILTER_MINIMUM = 3    # Remove at least 3 samples for the filtered correlation
MIN_OBSERVATIONS_FOR_CORR = 6  # Minimum samples required for a valid correlation

def calculate_p_value(t_stat: float, n: int) -> Optional[float]:
    """
    Calculates the two-tailed p-value for a given t-statistic and sample size.

    Args:
        t_stat : The t-statistic from the Pearson correlation.
        n      : The number of samples used in the correlation.

    Returns:
        The p-value, or None if the input is invalid (n < 3 or t_stat is NaN).
    """
    if t_stat is None or np.isnan(t_stat) or n < 3:
        return None
    # 2 * survival function (1 - cdf) of the t-distribution
    return 2 * t.sf(abs(t_stat), df=n - 2)

def calculate_weighted_stats(
    lf: pl.LazyFrame, 
    metric_col: str, 
    weight_col: str, 
    group_col: str,
    suffix: str
) -> pl.LazyFrame:
    """
    Calculates weighted Mean, Median, CV, P05, and P95 for a metric.
    """
    # 1. Weighted Mean
    mean_lf = lf.group_by(group_col).agg(
        (pl.col(metric_col) * pl.col(weight_col)).sum() / pl.col(weight_col).sum()
    ).rename({metric_col: f"mean_{suffix}"})

    # 2. Weighted Quantiles
    # Found by sorting and finding where the cumulative weight passes the threshold.
    quantiles_lf = (
        lf.sort(metric_col)
        .group_by(group_col)
        .agg([
            pl.col(metric_col).gather(
                pl.col(weight_col).cum_sum().search_sorted(pl.col(weight_col).sum() * 0.05)
            ).first().alias(f"p05_{suffix}"),
            pl.col(metric_col).gather(
                pl.col(weight_col).cum_sum().search_sorted(pl.col(weight_col).sum() * 0.50)
            ).first().alias(f"median_{suffix}"),
            pl.col(metric_col).gather(
                pl.col(weight_col).cum_sum().search_sorted(pl.col(weight_col).sum() * 0.95)
            ).first().alias(f"p95_{suffix}")
        ])
    )

    # 3. Weighted Standard Deviation & CV
    # Formula: sqrt( sum(w_i * (x_i - mean)^2) / (sum(w_i) - 1) )
    stdev_cv_lf = (
        lf.join(mean_lf, on=group_col)
        .group_by(group_col)
        .agg([
            (
                pl.when(pl.col(weight_col).sum() > 1)
                .then(
                    ((pl.col(weight_col) * (pl.col(metric_col) - pl.col(f"mean_{suffix}")).pow(2)).sum() /
                    (pl.col(weight_col).sum() - 1)).sqrt()
                )
                .otherwise(None)
            ).alias("stdev"),
            pl.col(f"mean_{suffix}").first().alias("mean")
        ])
        .with_columns(
            (pl.col("stdev") / pl.col("mean")).alias(f"cv_{suffix}")
        )
        .drop(["stdev", "mean"])
    )

    return mean_lf.join(quantiles_lf, on=group_col).join(stdev_cv_lf, on=group_col)

def produce_feature_kmer_global_logic(
    input_pattern: str,
    taxonomy_dir: Path,
    output_file: Path,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates the Grand Global k-mer feature generation.
    """
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace.")

    check_write_permission(output_file)

    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    # 1. Load Taxonomy
    click.secho(f"Loading JolTax taxonomy from {taxonomy_dir.name}...", fg="cyan", err=True)
    tree = JolTree.load(taxonomy_dir)
    
    # Pre-build depth and ancestor lookup data (using JolTree primitives)
    # entry/exit times allow O(1) ancestor/clade checks
    node_data = pl.DataFrame({
        "t_id": tree._index_to_id.astype(np.uint32),
        "depth": tree.depths.astype(np.uint8),
        "entry": tree.entry_times,
        "exit": tree.exit_times
    })

    # 2. Pool Data (Grand Totals)
    click.secho(f"Scanning and pooling k-mer data from '{input_pattern}'...", fg="cyan", err=True)
    
    # We aggregate ALL k-mers for each (t_id, kmer_hit) pair across all samples
    # We filter out 0 (unclassified) hits here as they are handled in grand_total
    grand_lf = (
        pl.scan_parquet(input_pattern)
        .group_by(["t_id", "kmer_tax_id"])
        .agg(pl.col("kmer_count").sum())
    )

    # 3. Label Hits (Clade vs Lineage vs Misclassified)
    click.secho("Phase 1/4: Labeling k-mer hits (clade vs lineage)...", fg="cyan", err=True)
    
    # Join with node data for both the target t_id and the k-mer hit
    labeled_lf = (
        grand_lf
        .join(node_data.lazy().rename({"t_id": "target_t_id", "depth": "root_depth", "entry": "root_entry", "exit": "root_exit"}), left_on="t_id", right_on="target_t_id")
        .join(node_data.lazy().rename({"t_id": "kmer_tax_id", "depth": "kmer_depth", "entry": "kmer_entry", "exit": "kmer_exit"}), on="kmer_tax_id")
        .with_columns([
            # is_in_clade: kmer hit is descendant of target (or target itself)
            ((pl.col("kmer_entry") >= pl.col("root_entry")) & (pl.col("kmer_exit") <= pl.col("root_exit"))).alias("is_in_clade"),
            # is_in_lineage: target is descendant of kmer hit (kmer hit is ancestor)
            ((pl.col("root_entry") >= pl.col("kmer_entry")) & (pl.col("root_exit") <= pl.col("kmer_exit"))).alias("is_in_lineage")
        ])
    )

    # 4. Calculate Core Counts & Ratios
    click.secho("Phase 2/4: Calculating grand totals and ratios...", fg="cyan", err=True)
    
    counts_lf = (
        labeled_lf
        .group_by("t_id")
        .agg([
            pl.col("kmer_count").filter(pl.col("is_in_clade")).sum().fill_null(0).alias("grand_clade_kmers"),
            pl.col("kmer_count").filter((~pl.col("is_in_clade")) & (pl.col("kmer_tax_id") > 0)).sum().fill_null(0).alias("grand_exclade_kmers"),
            pl.col("kmer_count").filter(pl.col("is_in_lineage") & (~pl.col("is_in_clade"))).sum().fill_null(0).alias("grand_lineage_kmers"),
            pl.col("kmer_count").filter(pl.col("kmer_tax_id") == 1).sum().fill_null(0).alias("grand_root_kmers"),
            pl.col("kmer_count").sum().alias("grand_total_kmers")
        ])
        .with_columns([
            (pl.col("grand_clade_kmers") + pl.col("grand_exclade_kmers")).alias("grand_classified_kmers"),
            (pl.col("grand_exclade_kmers") - pl.col("grand_lineage_kmers")).alias("grand_misclassified_kmers")
        ])
        .with_columns([
            (pl.col("grand_clade_kmers") / pl.col("grand_classified_kmers")).alias("grand_clade_to_classified_kmer_ratio"),
            (pl.col("grand_lineage_kmers") / pl.col("grand_classified_kmers")).alias("grand_lineage_to_classified_kmer_ratio"),
            (pl.col("grand_misclassified_kmers") / pl.col("grand_classified_kmers")).alias("grand_misclassified_to_classified_kmer_ratio"),
            (pl.when(pl.col("grand_misclassified_kmers") > 0)
             .then((pl.col("grand_clade_kmers") + pl.col("grand_lineage_kmers")) / pl.col("grand_misclassified_kmers"))
             .otherwise(1.0)).alias("grand_supporting_to_misclassified_kmer_ratio"),
            (pl.col("grand_clade_kmers") / pl.col("grand_total_kmers")).alias("grand_clade_to_total_kmer_ratio")
        ])
    )

    # 5. Calculate Weighted Distance/Depth Stats
    click.secho("Phase 3/4: Analyzing taxonomic distance and distribution...", fg="cyan", err=True)
    
    # To calculate distance and LCA depth, we need to join lineages (expensive)
    # BUT we only do this for the unique hits in the pooled data
    dist_base_lf = (
        labeled_lf
        .filter((~pl.col("is_in_clade")) & (pl.col("kmer_tax_id") > 0))
    )
    
    # Vectorized LCA Depth calculation using tree indices
    # Distance = (depth_A - depth_LCA) + (depth_B - depth_LCA)
    # Relative LCA Depth = depth_LCA / (depth_root - 1)
    
    # We'll calculate these in memory for the unique pairs (usually a small number)
    unique_pairs = dist_base_lf.select(["t_id", "kmer_tax_id"]).unique().collect()
    
    root_ids = unique_pairs["t_id"].to_numpy()
    kmer_ids = unique_pairs["kmer_tax_id"].to_numpy()
    
    lca_ids = tree.get_lca_batch(root_ids, kmer_ids)
    lca_depths = tree.depths[tree._get_indices(lca_ids)]
    
    pair_metrics_df = unique_pairs.with_columns([
        pl.Series("lca_depth", lca_depths.astype(np.uint8)),
        pl.Series("lca_id", lca_ids.astype(np.uint32))
    ]).join(
        node_data.rename({"t_id": "target_t_id", "depth": "root_depth"}).select(["target_t_id", "root_depth"]), 
        left_on="t_id",
        right_on="target_t_id"
    ).join(
        node_data.rename({"t_id": "k_tid", "depth": "k_depth"}).select(["k_tid", "k_depth"]),
        left_on="kmer_tax_id",
        right_on="k_tid"
    ).with_columns([
        ((pl.col("root_depth") - pl.col("lca_depth")) + (pl.col("k_depth") - pl.col("lca_depth"))).alias("distance"),
        (pl.col("lca_depth") / (pl.col("root_depth") - 1)).alias("relative_lca_depth"),
        (pl.col("k_depth") / (pl.col("root_depth") - 1)).alias("relative_k_depth")
    ])

    dist_lf = dist_base_lf.join(pair_metrics_df.lazy(), on=["t_id", "kmer_tax_id"])
    
    # Calculate weighted stats for the metrics
    dist_stats = calculate_weighted_stats(dist_lf, "distance", "kmer_count", "t_id", "grand_misclassified_kmer_distance")
    depth_stats = calculate_weighted_stats(dist_lf, "k_depth", "kmer_count", "t_id", "grand_misclassified_kmer_depth")
    lca_stats = calculate_weighted_stats(dist_lf, "relative_lca_depth", "kmer_count", "t_id", "grand_misclassified_kmer_relative_lca_depth")
    
    # Lineage-only stats
    lineage_dist_lf = dist_lf.filter(pl.col("is_in_lineage"))
    lineage_stats = calculate_weighted_stats(lineage_dist_lf, "relative_k_depth", "kmer_count", "t_id", "grand_lineage_kmer_relative_depth")

    # 6. Top Hits
    click.secho("Phase 4/4: Identifying top taxonomic competitors...", fg="cyan", err=True)
    
    misclassified_pooled = dist_lf.filter(~pl.col("is_in_lineage")).collect()
    
    top_hits = (
        misclassified_pooled
        .group_by(["t_id", "kmer_tax_id"])
        .agg(pl.col("kmer_count").sum())
        .sort("kmer_count", descending=True)
        .group_by("t_id")
        .agg([
            pl.col("kmer_tax_id").head(5).alias("grand_top_5_misclassified_kmer_tax_ids"),
            (pl.col("kmer_count").head(5) / pl.col("kmer_count").sum()).alias("grand_top_5_misclassified_kmer_shares")
        ])
    )
    
    # Map names for top hits in bulk
    all_top_ids = []
    for ids in top_hits["grand_top_5_misclassified_kmer_tax_ids"]:
        all_top_ids.extend(ids.to_list())
    
    if all_top_ids:
        names_df = tree.annotate(list(set(all_top_ids))).select(["t_id", "t_scientific_name"])
        names_map = dict(zip(names_df["t_id"].to_list(), names_df["t_scientific_name"].to_list()))
    else:
        names_map = {}
    
    top_hits = top_hits.with_columns(
        pl.col("grand_top_5_misclassified_kmer_tax_ids").map_elements(
            lambda ids: [names_map.get(tid, "Unknown") for tid in ids],
            return_dtype=pl.List(pl.String)
        ).alias("grand_top_5_misclassified_kmer_names")
    )

    # 7. Final Join & Save
    click.secho(f"Merging all features into {output_file.name}...", fg="cyan", err=True)
    
    final_df = (
        counts_lf.collect()
        .join(dist_stats.collect(), on="t_id", how="left")
        .join(depth_stats.collect(), on="t_id", how="left")
        .join(lca_stats.collect(), on="t_id", how="left")
        .join(lineage_stats.collect(), on="t_id", how="left")
        .join(top_hits, on="t_id", how="left")
        .sort("t_id")
    )

    # Flatten top hit lists to strings for CSV/TSV
    ext = output_file.suffix.lower()
    if ext != ".parquet":
        final_df = final_df.with_columns([
            pl.col("grand_top_5_misclassified_kmer_tax_ids").list.eval(pl.element().cast(pl.String)).list.join(";"),
            pl.col("grand_top_5_misclassified_kmer_names").list.join(";"),
            pl.col("grand_top_5_misclassified_kmer_shares").list.eval(pl.element().round(4).cast(pl.String)).list.join(";")
        ])
        if ext == ".tsv":
            final_df.write_csv(output_file, separator="\t")
        else:
            final_df.write_csv(output_file)
    else:
        final_df.write_parquet(output_file, compression="zstd")

    meta = get_standard_metadata(
        file_type="FEATURE_TABLE",
        source_path=Path(input_pattern).parent,
        compression="zstd" if ext == ".parquet" else "None",
        sorting="t_id",
        data_standard="MIXED"
    )
    save_companion_metadata(output_file, meta)

    return {
        "species_processed": final_df.height,
        "output_file": str(output_file)
    }

def generate_minimizer_correlations(
    df: pl.LazyFrame,
    inspect_df: pl.LazyFrame,
    tree,
    bad_samples_file: Optional[Path] = None
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Calculates species-level and clade-level minimizer coverage and correlation metrics.

    This engine validates taxonomic presence by comparing observed unique 
    minimizer coverage against an expected probabilistic model based on 
    sequencing depth.

    Args:
        df               : LazyFrame of merged reports (must have mm_tot, mm_uniq).
        inspect_df       : LazyFrame of Kraken inspect data (tax_id, clade_minimizers).
        tree             : Loaded JolTree instance for clade aggregation.
        bad_samples_file : Optional path to a file containing sample IDs to exclude.

    Returns:
        A tuple of (summary_df, long_df).
        summary_df: Wide-format DataFrame indexed by 't_id' with validation metrics.
        long_df: Long-format DataFrame with per-sample feature metrics.
    """
    
    # 1. Cleaning & Exclusions
    if bad_samples_file and bad_samples_file.exists():
        bad_samples = load_sample_id_list(bad_samples_file)
        click.secho(f"Excluding {len(bad_samples)} bad samples...", fg="yellow", err=True)
        df = df.filter(~pl.col("sample_id").is_in(bad_samples))

    # 2. Clade Aggregation
    # We must ensure we have clade-level read counts to match the clade-level minimizers.
    click.secho("Calculating cumulative clade counts...", fg="cyan", err=True)
    
    # Standardize sample_id to Categorical for consistent joining
    df = df.with_columns(pl.col("sample_id").cast(pl.Categorical))
    
    df_reads = calc_clade_sum(df.collect(), tree, min_reads=0, min_observed=0)
    
    # Re-join with original minimizer counts from the reports
    df_joined = (
        df_reads.lazy()
        .join(
            df.select(["t_id", "sample_id", "mm_tot", "mm_uniq"]),
            on=["t_id", "sample_id"],
            how="left"
        )
        .join(
            inspect_df.select([
                pl.col("tax_id").alias("t_id"), 
                pl.col("clade_minimizers").alias("mm_uniq_db")
            ]),
            on="t_id",
            how="left"
        )
    )

    # 3. Probabilistic Coverage Modeling
    # Algorithm: Observed Coverage (O) vs Expected Coverage (E).
    # E = 1 - (1 - 1/M)^R, where M is DB minimizers and R is reads.
    click.secho("Calculating expected unique minimizer coverage...", fg="cyan", err=True)
    
    feature_lf = (
        df_joined
        .filter(pl.col("t_id").is_in([UNCLASSIFIED_TID, FILTERED_TID]).not_())
        .with_columns([
            (pl.col("mm_uniq") / pl.col("mm_uniq_db")).alias("mm_obs_cov"),
            (pl.col("mm_uniq") / pl.col("clade_reads")).alias("_yield_per_read"),
            (pl.col("mm_uniq") / pl.col("mm_tot")).alias("mm_uniq_prop")
        ])
        .with_columns(
            norm_factor = pl.col("_yield_per_read").median().over("t_id")
        )
        .with_columns(
            mm_db_norm = pl.col("mm_uniq_db") / pl.col("norm_factor")
        )
        .with_columns(
            mm_exp_cov = 1 - (1 - (1 / pl.col("mm_db_norm"))).pow(pl.col("clade_reads"))
        )
        .rename({"_yield_per_read": "mm_uniq_per_read"})
    )

    # 4. Statistical Aggregation
    click.secho("Calculating Pearson correlations and distributional stats...", fg="cyan", err=True)

    x = pl.col("mm_obs_cov")
    y = pl.col("mm_exp_cov")
    n = pl.len()
    
    has_variance = (x.n_unique() > 1) & (y.n_unique() > 1)

    n_to_remove = (
        pl.max_horizontal(n * CORR_TOP_PERCENT_FILTER / 100, CORR_TOP_FILTER_MINIMUM)
        .round(0)
        .cast(pl.UInt32)
    )
    
    is_not_outlier = x.rank(method="ordinal", descending=True) > n_to_remove
    
    x_f = x.filter(is_not_outlier)
    y_f = y.filter(is_not_outlier)
    n_f = x_f.len()
    has_variance_f = (x_f.n_unique() > 1) & (y_f.n_unique() > 1)

    summary_df = (
        feature_lf
        .group_by("t_id")
        .agg([
            pl.when(n >= MIN_OBSERVATIONS_FOR_CORR).then(
                pl.when(has_variance).then(pl.corr(x, y)).otherwise(None)
            ).alias("mm_pearson_corr"),
            
            pl.when(n >= MIN_OBSERVATIONS_FOR_CORR).then(
                pl.when(has_variance).then(
                    pl.corr(x, y) * ((n - 2) / (1 - pl.corr(x, y).pow(2))).sqrt()
                ).otherwise(None)
            ).alias("_t_stat"),
            
            n.alias("mm_pearson_n"),

            pl.when(n >= MIN_OBSERVATIONS_FOR_CORR).then(
                pl.when(has_variance_f).then(pl.corr(x_f, y_f)).otherwise(None)
            ).alias("mm_pearson_filtered_corr"),

            pl.when(n >= MIN_OBSERVATIONS_FOR_CORR).then(
                pl.when(has_variance_f).then(
                    pl.corr(x_f, y_f) * ((n_f - 2) / (1 - pl.corr(x_f, y_f).pow(2))).sqrt()
                ).otherwise(None)
            ).alias("_t_stat_f"),

            n_f.alias("mm_pearson_filtered_n"),

            x.mean().alias("mm_obs_cov_mean"),
            x.median().alias("mm_obs_cov_median"),
            (x.std() / x.mean()).alias("mm_obs_cov_cv"),
            x.quantile(0.05).alias("mm_obs_cov_p05"),
            x.quantile(0.95).alias("mm_obs_cov_p95")
        ])
        .with_columns([
            pl.struct(["_t_stat", "mm_pearson_n"]).map_elements(
                lambda s: calculate_p_value(s["_t_stat"], s["mm_pearson_n"]),
                return_dtype=pl.Float64
            ).alias("mm_pearson_p"),
            
            pl.struct(["_t_stat_f", "mm_pearson_filtered_n"]).map_elements(
                lambda s: calculate_p_value(s["_t_stat_f"], s["mm_pearson_filtered_n"]),
                return_dtype=pl.Float64
            ).alias("mm_pearson_filtered_p")
        ])
        .drop(["_t_stat", "_t_stat_f"])
        .sort("t_id")
        .collect()
    )

    long_df = (
        feature_lf
        .select([
            pl.col("t_id"), 
            pl.col("sample_id"), 
            pl.col("clade_reads").alias("reads_tot"), 
            pl.col("mm_tot"), 
            pl.col("mm_uniq"),
            pl.col("mm_uniq_per_read"), 
            pl.col("mm_uniq_prop"), 
            pl.col("mm_obs_cov"), 
            pl.col("mm_exp_cov")
        ])
        .sort(["t_id", "sample_id"])
        .collect()
    )

    return summary_df, long_df

def produce_feature_uniq_minimizer_corr_logic(
    input_parquet: Path,
    inspect_csv: Path,
    taxonomy_dir: Path,
    output_file: Path,
    output_long_file: Optional[Path] = None,
    bad_samples_file: Optional[Path] = None,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates the minimizer correlation feature generation process.
    """
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace it.")
    
    if output_long_file and output_long_file.exists() and not overwrite:
        raise FileExistsError(f"Long-format output file '{output_long_file}' already exists. Use --overwrite to replace it.")

    check_write_permission(output_file)
    if output_long_file:
        check_write_permission(output_long_file)

    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    metadata = validate_sweetbits_file(
        input_parquet, 
        expected_type="REPORT_PARQUET",
        required_columns=["mm_tot", "mm_uniq"]
    )
    
    if metadata.get("report_format") != "HYPERLOGLOG":
        raise ValueError(
            f"Input file '{input_parquet.name}' was collected in LEGACY format. "
            "Minimizer correlations require HYPERLOGLOG reports (8-column format)."
        )

    click.secho(f"Loading JolTax taxonomy from {taxonomy_dir.name}...", fg="cyan", err=True)
    tree = JolTree.load(taxonomy_dir)

    click.secho("Loading provided Kraken inspect data...", fg="cyan", err=True)
    try:
        inspect_df = pl.scan_csv(inspect_csv, separator=";", schema_overrides={"tax_id": pl.UInt32})
        schema = inspect_df.collect_schema()
        if "clade_minimizers" not in schema:
            inspect_df = pl.scan_csv(inspect_csv, separator=",", schema_overrides={"tax_id": pl.UInt32})
    except Exception as e:
        raise ValueError(f"Failed to parse Kraken inspect file: {str(e)}")

    df = pl.scan_parquet(input_parquet)
    
    summary_df, long_df = generate_minimizer_correlations(
        df=df,
        inspect_df=inspect_df,
        tree=tree,
        bad_samples_file=bad_samples_file
    )

    click.secho(f"Saving results to {output_file.name}...", fg="cyan", err=True)
    
    ext = output_file.suffix.lower()
    if ext == ".parquet":
        summary_df.write_parquet(output_file, compression="zstd")
    elif ext == ".tsv":
        summary_df.write_csv(output_file, separator="\t")
    else:
        summary_df.write_csv(output_file)

    out_metadata = get_standard_metadata(
        file_type="FEATURE_TABLE",
        source_path=input_parquet,
        compression="None" if ext != ".parquet" else "zstd",
        sorting="t_id",
        data_standard=metadata.get("data_standard", "GENERIC"),
        report_format="HYPERLOGLOG"
    )
    save_companion_metadata(output_file, out_metadata)

    if output_long_file:
        click.secho(f"Saving long-format features to {output_long_file.name}...", fg="cyan", err=True)
        long_df.write_parquet(output_long_file, compression="zstd")
        
        long_metadata = get_standard_metadata(
            file_type="LONG_FEATURE_TABLE",
            source_path=input_parquet,
            compression="zstd",
            sorting="t_id, sample_id",
            data_standard=metadata.get("data_standard", "GENERIC"),
            report_format="HYPERLOGLOG"
        )
        save_companion_metadata(output_long_file, long_metadata)

    return {
        "taxa_processed": summary_df.height,
        "valid_correlations": summary_df.filter(pl.col("mm_pearson_corr").is_not_null()).height,
        "output_format": ext[1:].upper(),
        "long_format_saved": True if output_long_file else False
    }
