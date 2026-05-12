"""
sweetbits.features
Statistical and mathematical engines for metagenomic classification quality feature extraction.

This module provides four primary feature extraction engines:
1. Global K-mer Features: Pools all k-mer classification data across samples to 
   create a comprehensive evidence profile for every taxon.
2. Unique Minimizer Correlations: Validates taxonomic presence by comparing observed 
   minimizer coverage against a probabilistic expectation model.
3. Global Read Length Features: Pools all read length distributions to calculate 
   robust project-wide statistical metrics (Mean, Median, CV) per taxon.
4. Per-Sample Read Length Features: Generates high-resolution temporal timelines of 
   read length statistics for every taxon detected in each sample.
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
from sweetbits.metadata import validate_sweetbits_file, get_standard_metadata, save_companion_metadata, read_companion_metadata

# Parameters for the Minimizer Correlation engine
CORR_TOP_PERCENT_FILTER = 1.0  # Percentage of top coverage samples to remove for filtered correlation
CORR_TOP_FILTER_MINIMUM = 3    # Minimum number of samples to remove for filtered correlation
MIN_OBSERVATIONS_FOR_CORR = 6  # Minimum sample size (n) required to calculate a Pearson correlation

def calculate_p_value(t_stat: float, n: int) -> Optional[float]:
    """
    Calculates the two-tailed p-value for a Pearson correlation.

    Uses the survival function (1 - CDF) of the t-distribution with n-2 degrees of freedom.

    Args:
        t_stat : The t-statistic derived from the correlation coefficient and sample size.
        n      : The number of observations (samples).

    Returns:
        The p-value as a float, or None if calculation is not statistically valid (n < 3).
    """
    if t_stat is None or np.isnan(t_stat) or n < 3:
        return None
    return 2 * t.sf(abs(t_stat), df=n - 2)

def calculate_weighted_stats(
    lf: pl.LazyFrame, 
    metric_col: str, 
    weight_col: str, 
    group_col: Any,
    suffix: str
) -> pl.LazyFrame:
    """
    Calculates a suite of weighted distributional statistics for a given metric.

    This function computes the weighted Mean, Median, CV, 5th percentile, and 95th percentile.
    Weighting ensures that k-mer hits with higher counts have a proportional influence
    on the taxonomic distance and depth summaries.

    Args:
        lf         : LazyFrame containing the raw observations and weights.
        metric_col : The column to calculate statistics for (e.g., 'distance').
        weight_col : The column providing the weights (e.g., 'kmer_count').
        group_col  : The taxonomic key(s) to group by (e.g., 't_id' or ['t_id', 'sample_id']).
        suffix     : String suffix to append to the output column names.

    Returns:
        A LazyFrame with one row per grouping and the calculated statistical columns.
    """
    # 1. Weighted Mean
    # Calculated as sum(value * weight) / sum(weight)
    # We cast value to Float64 before multiplication to prevent 32-bit integer overflow
    # with high read counts (e.g. 30M reads * 150bp).
    mean_lf = lf.group_by(group_col).agg(
        ((pl.col(metric_col).cast(pl.Float64) * pl.col(weight_col)).sum() / pl.col(weight_col).sum()).alias(f"{suffix}_mean")
    )

    # 2. Weighted Quantiles
    # Determined by sorting the observations and identifying the values where the 
    # cumulative weight crosses the 5%, 50% (median), and 95% thresholds.
    quantiles_lf = (
        lf.sort(metric_col)
        .group_by(group_col)
        .agg([
            pl.col(metric_col).gather(
                pl.col(weight_col).cum_sum().search_sorted(pl.col(weight_col).sum() * 0.05)
            ).first().alias(f"{suffix}_p05"),
            pl.col(metric_col).gather(
                pl.col(weight_col).cum_sum().search_sorted(pl.col(weight_col).sum() * 0.50)
            ).first().alias(f"{suffix}_median"),
            pl.col(metric_col).gather(
                pl.col(weight_col).cum_sum().search_sorted(pl.col(weight_col).sum() * 0.95)
            ).first().alias(f"{suffix}_p95")
        ])
    )

    # 3. Weighted Standard Deviation & Coefficient of Variation (CV)
    # Variance formula: sum(w_i * (x_i - mean)^2) / (sum(w_i) - 1)
    stdev_cv_lf = (
        lf.join(mean_lf, on=group_col)
        .group_by(group_col)
        .agg([
            (
                pl.when(pl.col(weight_col).sum() > 1)
                .then(
                    ((pl.col(weight_col) * (pl.col(metric_col) - pl.col(f"{suffix}_mean")).pow(2)).sum() /
                    (pl.col(weight_col).sum() - 1)).sqrt()
                )
                .otherwise(None)
            ).alias(f"{suffix}_stdev"),
            pl.col(f"{suffix}_mean").first().alias("mean")
        ])
        .with_columns(
            pl.when(pl.col("mean") != 0)
            .then(pl.col(f"{suffix}_stdev") / pl.col("mean"))
            .otherwise(None)
            .alias(f"{suffix}_cv")
        )
        .drop("mean")
    )

    res_lf = mean_lf.join(quantiles_lf, on=group_col).join(stdev_cv_lf, on=group_col)

    # Standardize column order
    return res_lf.select([
        *([group_col] if isinstance(group_col, str) else group_col),
        f"{suffix}_mean",
        f"{suffix}_median",
        f"{suffix}_stdev",
        f"{suffix}_cv",
        f"{suffix}_p05",
        f"{suffix}_p95"
    ])

def produce_feature_kmer_global_logic(
    input_pattern: str,
    taxonomy_dir: Path,
    output_file: Path,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates the Global k-mer feature generation engine.

    This engine operates in four distinct phases:
    1. Labeling: Every k-mer hit is categorized as 'Clade', 'Lineage', or 'Misclassified' 
       using a Nested Set Model for O(1) taxonomic checks.
    2. Aggregation: 14 distinct ratios and confidence scores are calculated for 
       every species rank taxon.
    3. Distributional Analysis: Weighted distances and depths are calculated to 
       quantify the "biological closeness" of off-target noise.
    4. Competition Profiling: Top 5 taxonomic competitors are identified for 
       each species based on misclassification volume.

    Args:
        input_pattern : Glob pattern for the ingested .kmers.parquet files.
        taxonomy_dir  : Path to the JolTax cache directory.
        output_file   : Path to save the resulting feature table (CSV, TSV, or Parquet).
        cores         : Number of threads to dedicate to Polars.
        overwrite     : Whether to overwrite an existing output file.

    Returns:
        A dictionary containing processing statistics.
    """
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace.")

    check_write_permission(output_file)

    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    # 1. Load Taxonomy
    click.secho(f"Loading JolTax taxonomy from {taxonomy_dir.name}...", fg="cyan", err=True)
    tree = JolTree.load(taxonomy_dir)
    
    # Pre-build a lightweight taxonomic reference table.
    # The 'entry' and 'exit' times allow us to determine if one node is a descendant
    # of another without traversing the tree, enabling high-speed vectorized labeling.
    node_data = pl.DataFrame({
        "t_id": tree._index_to_id.astype(np.uint32),
        "depth": tree.depths.astype(np.uint8),
        "entry": tree.entry_times,
        "exit": tree.exit_times
    })

    # 2. Pool Data (Global Totals)
    # Scans all sample summaries and groups by (target_taxon, kmer_hit).
    # TaxID 0 is explicitly kept here to support global total and unclassified ratio math.
    click.secho(f"Scanning and pooling k-mer data from '{input_pattern}'...", fg="cyan", err=True)
    
    global_lf = (
        pl.scan_parquet(input_pattern)
        .group_by(["t_id", "kmer_tax_id"])
        .agg(pl.col("kmer_count").cast(pl.UInt64).sum())
    )

    # 3. Label Hits (Clade vs Lineage vs Misclassified)
    # Phase 1/4: Identifies the relationship between the Kraken assignment and the k-mer evidence.
    click.secho("Phase 1/4: Labeling all k-mer hits (clade vs lineage vs misclassified)...", fg="cyan", err=True)
    
    # Materialize the calculation 'trunk' into memory here. 
    # This summarized table is small enough to fit in RAM but ensures that the 
    # massive project-wide Parquet scan happens exactly once.
    labeled_eager_df = (
        global_lf
        .join(node_data.lazy().rename({"t_id": "target_t_id", "depth": "root_depth", "entry": "root_entry", "exit": "root_exit"}), left_on="t_id", right_on="target_t_id", how="left")
        .join(node_data.lazy().rename({"t_id": "kmer_tax_id", "depth": "kmer_depth", "entry": "kmer_entry", "exit": "kmer_exit"}), on="kmer_tax_id", how="left")
        .with_columns([
            # is_in_clade: kmer hit is descendant of target (or target itself)
            ((pl.col("kmer_entry") >= pl.col("root_entry")) & (pl.col("kmer_exit") <= pl.col("root_exit"))).fill_null(False).alias("is_in_clade"),
            # is_in_lineage: target is descendant of kmer hit (kmer hit is ancestor)
            ((pl.col("root_entry") >= pl.col("kmer_entry")) & (pl.col("root_exit") <= pl.col("kmer_exit"))).fill_null(False).alias("is_in_lineage")
        ])
    ).collect()
    
    # All subsequent operations branch from this in-memory summarized source.
    labeled_lf = labeled_eager_df.lazy()

    # 4. Calculate Core Counts & Ratios
    # Phase 2/4: Calculates absolute counts and high-signal quality ratios.
    click.secho("Phase 2/4: Calculating global k-mer totals and ratios...", fg="cyan", err=True)
    
    def safe_div(num_expr, den_expr):
        return pl.when(den_expr > 0).then(num_expr / den_expr).otherwise(None)

    counts_lf = (
        labeled_lf
        .group_by("t_id")
        .agg([
            pl.col("kmer_count").filter(pl.col("is_in_clade")).sum().fill_null(0).alias("kmers_global_clade_count"),
            pl.col("kmer_count").filter((~pl.col("is_in_clade")) & (pl.col("kmer_tax_id") > 0)).sum().fill_null(0).alias("kmers_global_exclade_count"),
            pl.col("kmer_count").filter(pl.col("is_in_lineage") & (~pl.col("is_in_clade"))).sum().fill_null(0).alias("kmers_global_lineage_count"),
            pl.col("kmer_count").filter((pl.col("kmer_tax_id") == 1) & (~pl.col("is_in_clade"))).sum().fill_null(0).alias("kmers_global_root_count"),
            pl.col("kmer_count").sum().alias("kmers_global_total_count")
        ])
        .with_columns([
            (pl.col("kmers_global_clade_count") + pl.col("kmers_global_exclade_count")).alias("kmers_global_classified_count"),
            (pl.col("kmers_global_exclade_count") - pl.col("kmers_global_lineage_count")).alias("kmers_global_misclassified_count")
        ])
        .with_columns([
            (pl.col("kmers_global_total_count") - pl.col("kmers_global_classified_count")).alias("kmers_global_unclassified_count")
        ])
        .with_columns([
            safe_div(pl.col("kmers_global_clade_count"), pl.col("kmers_global_classified_count")).alias("kmers_global_cladeVSclassified_ratio"),
            safe_div(pl.col("kmers_global_lineage_count"), pl.col("kmers_global_classified_count")).alias("kmers_global_lineageVSclassified_ratio"),
            safe_div(pl.col("kmers_global_misclassified_count"), pl.col("kmers_global_classified_count")).alias("kmers_global_misclassifiedVSclassified_ratio"),
            safe_div(pl.col("kmers_global_root_count"), pl.col("kmers_global_classified_count")).alias("kmers_global_rootVSclassified_ratio"),
            
            safe_div(pl.col("kmers_global_clade_count"), pl.col("kmers_global_total_count")).alias("kmers_global_cladeVStotal_ratio"),
            safe_div(pl.col("kmers_global_classified_count"), pl.col("kmers_global_total_count")).alias("kmers_global_classifiedVStotal_ratio"),
            safe_div(pl.col("kmers_global_lineage_count"), pl.col("kmers_global_total_count")).alias("kmers_global_lineageVStotal_ratio"),
            safe_div(pl.col("kmers_global_root_count"), pl.col("kmers_global_total_count")).alias("kmers_global_rootVStotal_ratio"),
            safe_div(pl.col("kmers_global_misclassified_count"), pl.col("kmers_global_total_count")).alias("kmers_global_misclassifiedVStotal_ratio"),
            safe_div(pl.col("kmers_global_exclade_count"), pl.col("kmers_global_total_count")).alias("kmers_global_excladeVStotal_ratio"),
            safe_div((pl.col("kmers_global_clade_count") + pl.col("kmers_global_lineage_count")), pl.col("kmers_global_total_count")).alias("kmers_global_supportingVStotal_ratio"),
            
            safe_div(pl.col("kmers_global_root_count"), pl.col("kmers_global_exclade_count")).alias("kmers_global_rootVSexclade_ratio"),
            safe_div(pl.col("kmers_global_lineage_count"), pl.col("kmers_global_exclade_count")).alias("kmers_global_lineageVSexclade_ratio"),
        ])
    )

    # 5. Calculate Weighted Distance/Depth Stats
    # Phase 3/4: Measures the "taxonomic distance" of k-mer noise from the target species.
    click.secho("Phase 3/4: Analyzing taxonomic distance and distribution of out-of-clade k-mers...", fg="cyan", err=True)
    
    # Distance = (target_depth - LCA_depth) + (hit_depth - LCA_depth)
    # Relative LCA Depth = LCA_depth / (target_depth - 1)
    
    dist_base_lf = (
        labeled_lf
        .filter((~pl.col("is_in_clade")) & (pl.col("kmer_tax_id") > 0))
    )
    
    # LCA calculations are computationally intensive, so we compute them once for unique (target, hit) pairs.
    unique_pairs = dist_base_lf.select(["t_id", "kmer_tax_id"]).unique().collect()
    
    root_ids = unique_pairs["t_id"].to_numpy()
    kmer_ids = unique_pairs["kmer_tax_id"].to_numpy()
    
    lca_ids = tree.get_lca_batch(root_ids, kmer_ids)
    lca_depths = tree.depths[tree._get_indices(lca_ids)]
    
    pair_metrics_df = unique_pairs.with_columns([
        pl.Series("lca_depth", lca_depths.astype(np.uint8)),
        pl.Series("lca_id", lca_ids.astype(np.uint32))
    ])

    # Final distance and relative metrics. Denominators are cast to Int32 to prevent UInt8 underflow.
    dist_lf = dist_base_lf.join(pair_metrics_df.lazy(), on=["t_id", "kmer_tax_id"]).with_columns([
        ((pl.col("root_depth") - pl.col("lca_depth")) + (pl.col("kmer_depth") - pl.col("lca_depth"))).alias("distance"),
        (pl.col("lca_depth") / (pl.col("root_depth").cast(pl.Int32) - 1)).alias("relative_lca_depth"),
        (pl.col("kmer_depth") / (pl.col("root_depth").cast(pl.Int32) - 1)).alias("relative_k_depth")
    ])
    
    # Isolate strictly misclassified k-mers (hits outside the species lineage)
    misclassified_dist_lf = dist_lf.filter(~pl.col("is_in_lineage"))
    
    # Weighted statistics quantify the 'noise profile' for each species.
    dist_stats = calculate_weighted_stats(misclassified_dist_lf, "distance", "kmer_count", "t_id", "kmers_global_misclassified_dist")
    depth_stats = calculate_weighted_stats(misclassified_dist_lf, "kmer_depth", "kmer_count", "t_id", "kmers_global_misclassified_depth")
    lca_stats = calculate_weighted_stats(misclassified_dist_lf, "relative_lca_depth", "kmer_count", "t_id", "kmers_global_misclassified_relative_lca_depth")
    
    # Lineage-only statistics (hits between assigned species and root)
    lineage_dist_lf = dist_lf.filter(pl.col("is_in_lineage"))
    lineage_stats = calculate_weighted_stats(lineage_dist_lf, "relative_k_depth", "kmer_count", "t_id", "kmers_global_lineage_relative_depth")

    # 6. Top Hits
    # Phase 4/4: Profiles the primary taxonomic competitors for each identification.
    click.secho("Phase 4/4: Identifying top taxa for out-of-clade k-mer hits...", fg="cyan", err=True)
    
    exclade_pooled = dist_lf.collect()
    misclassified_pooled = exclade_pooled.filter(~pl.col("is_in_lineage"))
    
    # Top hits are sorted by count (Descending) and TaxID (Ascending) for deterministic tie-breaking.
    top_hits = (
        misclassified_pooled
        .group_by("t_id")
        .agg([
            pl.col("kmer_tax_id").sort_by(["kmer_count", "kmer_tax_id"], descending=[True, False]).head(5).alias("kmers_global_misclassified_top5_taxids"),
            (pl.col("kmer_count").sort_by(["kmer_count", "kmer_tax_id"], descending=[True, False]).head(5) / pl.col("kmer_count").sum()).alias("kmers_global_misclassified_top5_shares")
        ])
    ).join(
        exclade_pooled
        .group_by("t_id")
        .agg([
            pl.col("kmer_tax_id").sort_by(["kmer_count", "kmer_tax_id"], descending=[True, False]).head(5).alias("kmers_global_exclade_top5_taxids"),
            (pl.col("kmer_count").sort_by(["kmer_count", "kmer_tax_id"], descending=[True, False]).head(5) / pl.col("kmer_count").sum()).alias("kmers_global_exclade_top5_shares")
        ]), on="t_id", how="full", coalesce=True
    )
    
    # Resolve scientific names for the competitor TaxIDs in a single batch operation.
    all_top_ids = top_hits["kmers_global_misclassified_top5_taxids"].explode().drop_nulls().unique().to_list()
    exclade_ids = top_hits["kmers_global_exclade_top5_taxids"].explode().drop_nulls().unique().to_list()
    all_top_ids = list(set(all_top_ids + exclade_ids))
    
    if all_top_ids:
        names_df = tree.annotate(list(set(all_top_ids))).select(["t_id", "t_scientific_name"])
        names_map = dict(zip(names_df["t_id"].to_list(), names_df["t_scientific_name"].to_list()))
    else:
        names_map = {}
    
    top_hits = top_hits.with_columns([
        pl.col("kmers_global_misclassified_top5_taxids").map_elements(
            lambda ids: [names_map.get(tid, "Unknown") for tid in ids] if ids is not None else None,
            return_dtype=pl.List(pl.String)
        ).alias("kmers_global_misclassified_top5_names"),
        pl.col("kmers_global_exclade_top5_taxids").map_elements(
            lambda ids: [names_map.get(tid, "Unknown") for tid in ids] if ids is not None else None,
            return_dtype=pl.List(pl.String)
        ).alias("kmers_global_exclade_top5_names")
    ])

    # 7. Final Join & Save
    click.secho(f"Merging all features into {output_file.name}...", fg="cyan", err=True)
    
    # We maintain a strictly ordered schema for readability
    counts_cols = [
        "kmers_global_total_count",
        "kmers_global_classified_count",
        "kmers_global_unclassified_count",
        "kmers_global_clade_count",
        "kmers_global_exclade_count",
        "kmers_global_lineage_count",
        "kmers_global_root_count"
    ]
    ratio_cols = [
        "kmers_global_cladeVSclassified_ratio",
        "kmers_global_lineageVSclassified_ratio",
        "kmers_global_misclassifiedVSclassified_ratio",
        "kmers_global_rootVSclassified_ratio",
        "kmers_global_cladeVStotal_ratio",
        "kmers_global_classifiedVStotal_ratio",
        "kmers_global_lineageVStotal_ratio",
        "kmers_global_rootVStotal_ratio",
        "kmers_global_misclassifiedVStotal_ratio",
        "kmers_global_excladeVStotal_ratio",
        "kmers_global_supportingVStotal_ratio",
        "kmers_global_rootVSexclade_ratio",
        "kmers_global_lineageVSexclade_ratio"
    ]
    
    final_df = (
        counts_lf.collect()
        .join(dist_stats.collect(), on="t_id", how="left")
        .join(depth_stats.collect(), on="t_id", how="left")
        .join(lca_stats.collect(), on="t_id", how="left")
        .join(lineage_stats.collect(), on="t_id", how="left")
        .join(top_hits, on="t_id", how="left")
    )

    # Get structural stat columns from the joined frames (they are already ordered by calculate_weighted_stats)
    struct_cols = []
    for suffix in ["kmers_global_misclassified_dist", "kmers_global_misclassified_depth", "kmers_global_misclassified_relative_lca_depth", "kmers_global_lineage_relative_depth"]:
        struct_cols.extend([f"{suffix}_{m}" for f in [suffix] for m in ["mean", "median", "stdev", "cv", "p05", "p95"]])

    top_cols = [c for c in top_hits.columns if c != "t_id"]

    final_df = final_df.select([
        "t_id",
        *counts_cols,
        *ratio_cols,
        *struct_cols,
        *top_cols
    ]).sort("t_id")

    # Convert list columns to character-delimited strings.
    # Non-numeric columns are explicitly quoted for parsing safety in downstream tools.
    ext = output_file.suffix.lower()
    if ext != ".parquet":
        final_df = final_df.with_columns([
            pl.col("kmers_global_misclassified_top5_taxids").list.eval(pl.element().cast(pl.String)).list.join(";"),
            pl.col("kmers_global_misclassified_top5_names").list.join(";"),
            pl.col("kmers_global_misclassified_top5_shares").list.eval(pl.element().round(4).cast(pl.String)).list.join(";"),
            pl.col("kmers_global_exclade_top5_taxids").list.eval(pl.element().cast(pl.String)).list.join(";"),
            pl.col("kmers_global_exclade_top5_names").list.join(";"),
            pl.col("kmers_global_exclade_top5_shares").list.eval(pl.element().round(4).cast(pl.String)).list.join(";")
        ])
        if ext == ".tsv":
            final_df.write_csv(output_file, separator="\t", quote_style="non_numeric")
        else:
            final_df.write_csv(output_file, quote_style="non_numeric")
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

    Algorithm:
    1. Observed Coverage (O): (Observed Unique Minimizers) / (DB Unique Minimizers)
    2. Expected Coverage (E): 1 - (1 - 1/M)^R, where M is normalized DB minimizers 
       and R is the number of reads assigned to the clade.

    A high correlation between O and E across samples indicates that the 
    evidence for the taxon is accumulating in a biologically plausible manner.

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
    # Minimizer metrics are typically reported for clades, so read counts must 
    # be rolled up to the clade level before modeling.
    click.secho("Calculating cumulative clade counts...", fg="cyan", err=True)
    
    df = df.with_columns(pl.col("sample_id").cast(pl.Categorical))
    
    df_reads = calc_clade_sum(df.collect(), tree, min_reads=0, min_observed=0)
    
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
    # Phase 1: Simple Pearson Correlation.
    # Phase 2: Filtered Correlation (Outlier rejection) to handle anomalous depth spikes.
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
            x.std().alias("mm_obs_cov_stdev"),
            pl.when(x.mean() != 0).then(x.std() / x.mean()).otherwise(None).alias("mm_obs_cov_cv"),
            x.quantile(0.05, interpolation="nearest").alias("mm_obs_cov_p05"),
            x.quantile(0.95, interpolation="nearest").alias("mm_obs_cov_p95")
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
        .select([
            "t_id",
            "mm_pearson_corr",
            "mm_pearson_p",
            "mm_pearson_n",
            "mm_pearson_filtered_corr",
            "mm_pearson_filtered_p",
            "mm_pearson_filtered_n",
            "mm_obs_cov_mean",
            "mm_obs_cov_median",
            "mm_obs_cov_stdev",
            "mm_obs_cov_cv",
            "mm_obs_cov_p05",
            "mm_obs_cov_p95"
        ])
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
    Orchestrates the unique minimizer correlation feature generation logic.

    This high-level function handles environment setup, metadata validation, 
    taxonomy loading, and CSV/Parquet output generation.
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
        summary_df.write_csv(output_file, separator="\t", quote_style="non_numeric")
    else:
        summary_df.write_csv(output_file, quote_style="non_numeric")

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


def produce_feature_read_lengths_global_logic(
    input_pattern: str,
    output_file: Path,
    min_reads: int = 50,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates the Global read length feature generation engine.

    This engine pools read length distributions across all samples to calculate
    robust project-wide statistical metrics (Mean, Median, p05, p95, CV) for every
    taxon that meets the minimum read threshold.

    Args:
        input_pattern : Glob pattern for the ingested .read_lengths.parquet files.
        output_file   : Path to save the global feature table (CSV, TSV, or Parquet).
        min_reads     : Minimum total reads required for a taxon to be included.
        cores         : Number of threads to dedicate to Polars.
        overwrite     : Whether to overwrite an existing output file.

    Returns:
        A dictionary containing processing statistics.
    """
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace.")

    check_write_permission(output_file)

    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    click.secho(f"Scanning and pooling read lengths from '{input_pattern}'...", fg="cyan", err=True)
    
    base_lf = pl.scan_parquet(input_pattern)
    
    # 1. Pool counts across all samples
    pooled_lf = (
        base_lf
        .group_by(["t_id", "read_length"])
        .agg(pl.col("read_count").sum())
    )
    
    # 2. Identify taxa passing the threshold
    totals_lf = (
        pooled_lf
        .group_by("t_id")
        .agg(pl.col("read_count").sum().alias("reads_global_total_count"))
        .filter(pl.col("reads_global_total_count") >= min_reads)
    )
    
    # 3. Calculate weighted statistics
    stats_lf = calculate_weighted_stats(
        pooled_lf.join(totals_lf.select("t_id"), on="t_id", how="inner"),
        metric_col="read_length",
        weight_col="read_count",
        group_col="t_id",
        suffix="reads_global_readlen"
    ).join(totals_lf, on="t_id")

    # Standardize column order
    df = stats_lf.select([
        "t_id",
        "reads_global_total_count",
        "reads_global_readlen_mean",
        "reads_global_readlen_median",
        "reads_global_readlen_stdev",
        "reads_global_readlen_cv",
        "reads_global_readlen_p05",
        "reads_global_readlen_p95"
    ]).collect(engine="streaming").sort("t_id")

    # 4. Save and Metadata
    click.secho(f"Saving global results to {output_file.name}...", fg="cyan", err=True)
    ext = output_file.suffix.lower()
    if ext == ".parquet":
        df.write_parquet(output_file, compression="zstd")
    elif ext == ".tsv":
        df.write_csv(output_file, separator="\t", quote_style="non_numeric")
    else:
        df.write_csv(output_file, quote_style="non_numeric")

    # Detect data standard from metadata
    first_file = list(Path(input_pattern).parent.glob(Path(input_pattern).name))[0]
    from sweetbits.metadata import read_companion_metadata
    meta = read_companion_metadata(first_file) or {}
    
    out_meta = get_standard_metadata(
        file_type="FEATURE_TABLE",
        source_path=Path(input_pattern).parent,
        compression="zstd" if ext == ".parquet" else "None",
        sorting="t_id",
        data_standard=meta.get("data_standard", "GENERIC")
    )
    save_companion_metadata(output_file, out_meta)

    return {
        "taxa_processed": df.height,
        "min_reads_filter": min_reads,
        "output_file": str(output_file)
    }

def produce_feature_read_lengths_sample_logic(
    input_pattern: str,
    output_file: Path,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates the Per-Sample read length feature generation engine.

    This engine calculates weighted read length statistics for every taxon detected
    in each individual sample. No minimum read filter is applied to ensure 
    complete temporal timelines for visualization.

    Args:
        input_pattern : Glob pattern for the ingested .read_lengths.parquet files.
        output_file   : Path to save the long-format parquet table.
        cores         : Number of threads to dedicate to Polars.
        overwrite     : Whether to overwrite an existing output file.

    Returns:
        A dictionary containing processing statistics.
    """
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace.")

    check_write_permission(output_file)

    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    click.secho(f"Calculating per-sample read length statistics from '{input_pattern}'...", fg="cyan", err=True)
    
    base_lf = pl.scan_parquet(input_pattern)
    schema = base_lf.collect_schema()
    
    group_cols = ["t_id", "sample_id"]
    if "year" in schema:
        group_cols.append("year")
    if "week" in schema:
        group_cols.append("week")

    # 1. Calculate weighted statistics
    stats_lf = calculate_weighted_stats(
        base_lf, 
        metric_col="read_length", 
        weight_col="read_count", 
        group_col=group_cols, 
        suffix="reads_sample_readlen"
    ).join(
        base_lf.group_by(group_cols).agg(pl.col("read_count").sum().alias("reads_sample_total_count")),
        on=group_cols
    )

    # Standardize column order
    df = stats_lf.select([
        *group_cols,
        "reads_sample_total_count",
        "reads_sample_readlen_mean",
        "reads_sample_readlen_median",
        "reads_sample_readlen_stdev",
        "reads_sample_readlen_cv",
        "reads_sample_readlen_p05",
        "reads_sample_readlen_p95"
    ]).collect(engine="streaming").sort(["t_id", "sample_id"])

    # 2. Save and Metadata
    click.secho(f"Saving per-sample results to {output_file.name}...", fg="cyan", err=True)
    df.write_parquet(output_file, compression="zstd")

    first_file = list(Path(input_pattern).parent.glob(Path(input_pattern).name))[0]
    from sweetbits.metadata import read_companion_metadata
    meta = read_companion_metadata(first_file) or {}
    
    out_meta = get_standard_metadata(
        file_type="READ_LEN_FEATURE_LONG_PARQUET",
        source_path=Path(input_pattern).parent,
        compression="zstd",
        sorting="t_id, sample_id",
        data_standard=meta.get("data_standard", "GENERIC")
    )
    save_companion_metadata(output_file, out_meta)

    return {
        "records_processed": df.height,
        "output_file": str(output_file)
    }


def produce_feature_kmer_sample_logic(
    input_pattern: str,
    taxonomy_dir: Path,
    output_file: Path,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates the Per-Sample k-mer feature generation engine.

    This engine operates in four phases (mirroring the global engine):
    1. Labeling: K-mers are categorized as 'Clade', 'Lineage', or 'Misclassified'.
    2. Aggregation: 13 distinct ratios and confidence scores are calculated per species PER SAMPLE.
    3. Distributional Analysis: Weighted distances and depths are calculated.
    4. Competition Profiling: Top 3 taxonomic competitors are identified per sample.

    Args:
        input_pattern : Glob pattern for the ingested .kmers.parquet files.
        taxonomy_dir  : Path to the JolTax cache directory.
        output_file   : Path to save the resulting long-format Parquet file.
        cores         : Number of threads to dedicate to Polars.
        overwrite     : Whether to overwrite an existing output file.

    Returns:
        A dictionary containing processing statistics.
    """
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace.")

    check_write_permission(output_file)

    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    # 1. Load Taxonomy
    click.secho(f"Loading JolTax taxonomy from {taxonomy_dir.name}...", fg="cyan", err=True)
    tree = JolTree.load(taxonomy_dir)
    
    node_data = pl.DataFrame({
        "t_id": tree._index_to_id.astype(np.uint32),
        "depth": tree.depths.astype(np.uint8),
        "entry": tree.entry_times,
        "exit": tree.exit_times
    })

    # 2. Pool Data (Sample Resolution)
    click.secho(f"Scanning k-mer data from '{input_pattern}'...", fg="cyan", err=True)
    
    # We maintain sample_id grouping
    base_lf = pl.scan_parquet(input_pattern)

    # 3. Label Hits
    click.secho("Phase 1/4: Labeling k-mer hits per sample...", fg="cyan", err=True)
    
    labeled_eager_df = (
        base_lf
        .join(node_data.lazy().rename({"t_id": "target_t_id", "depth": "root_depth", "entry": "root_entry", "exit": "root_exit"}), left_on="t_id", right_on="target_t_id", how="left")
        .join(node_data.lazy().rename({"t_id": "kmer_tax_id", "depth": "kmer_depth", "entry": "kmer_entry", "exit": "kmer_exit"}), on="kmer_tax_id", how="left")
        .with_columns([
            ((pl.col("kmer_entry") >= pl.col("root_entry")) & (pl.col("kmer_exit") <= pl.col("root_exit"))).fill_null(False).alias("is_in_clade"),
            ((pl.col("root_entry") >= pl.col("kmer_entry")) & (pl.col("root_exit") <= pl.col("kmer_exit"))).fill_null(False).alias("is_in_lineage")
        ])
    ).collect()
    
    labeled_lf = labeled_eager_df.lazy()

    # 4. Calculate Core Counts & Ratios
    click.secho("Phase 2/4: Calculating per-sample k-mer totals and ratios...", fg="cyan", err=True)
    
    # We define a helper for safe division (yields Null if denominator is 0)
    def safe_div(num_expr, den_expr):
        return pl.when(den_expr > 0).then(num_expr / den_expr).otherwise(None)

    counts_lf = (
        labeled_lf
        .group_by(["sample_id", "t_id"])
        .agg([
            pl.col("kmer_count").filter(pl.col("is_in_clade")).sum().fill_null(0).alias("kmers_sample_clade_count"),
            pl.col("kmer_count").filter((~pl.col("is_in_clade")) & (pl.col("kmer_tax_id") > 0)).sum().fill_null(0).alias("kmers_sample_exclade_count"),
            pl.col("kmer_count").filter(pl.col("is_in_lineage") & (~pl.col("is_in_clade"))).sum().fill_null(0).alias("kmers_sample_lineage_count"),
            pl.col("kmer_count").filter((pl.col("kmer_tax_id") == 1) & (~pl.col("is_in_clade"))).sum().fill_null(0).alias("kmers_sample_root_count"),
            pl.col("kmer_count").sum().alias("kmers_sample_total_count")
        ])
        .with_columns([
            (pl.col("kmers_sample_clade_count") + pl.col("kmers_sample_exclade_count")).alias("kmers_sample_classified_count"),
            (pl.col("kmers_sample_exclade_count") - pl.col("kmers_sample_lineage_count")).alias("kmers_sample_misclassified_count")
        ])
        .with_columns([
            (pl.col("kmers_sample_total_count") - pl.col("kmers_sample_classified_count")).alias("kmers_sample_unclassified_count")
        ])
        .with_columns([
            safe_div(pl.col("kmers_sample_clade_count"), pl.col("kmers_sample_classified_count")).alias("kmers_sample_cladeVSclassified_ratio"),
            safe_div(pl.col("kmers_sample_lineage_count"), pl.col("kmers_sample_classified_count")).alias("kmers_sample_lineageVSclassified_ratio"),
            safe_div(pl.col("kmers_sample_misclassified_count"), pl.col("kmers_sample_classified_count")).alias("kmers_sample_misclassifiedVSclassified_ratio"),
            safe_div(pl.col("kmers_sample_root_count"), pl.col("kmers_sample_classified_count")).alias("kmers_sample_rootVSclassified_ratio"),
            
            safe_div(pl.col("kmers_sample_clade_count"), pl.col("kmers_sample_total_count")).alias("kmers_sample_cladeVStotal_ratio"),
            safe_div(pl.col("kmers_sample_classified_count"), pl.col("kmers_sample_total_count")).alias("kmers_sample_classifiedVStotal_ratio"),
            safe_div(pl.col("kmers_sample_lineage_count"), pl.col("kmers_sample_total_count")).alias("kmers_sample_lineageVStotal_ratio"),
            safe_div(pl.col("kmers_sample_root_count"), pl.col("kmers_sample_total_count")).alias("kmers_sample_rootVStotal_ratio"),
            safe_div(pl.col("kmers_sample_misclassified_count"), pl.col("kmers_sample_total_count")).alias("kmers_sample_misclassifiedVStotal_ratio"),
            safe_div((pl.col("kmers_sample_clade_count") + pl.col("kmers_sample_lineage_count")), pl.col("kmers_sample_total_count")).alias("kmers_sample_supportingVStotal_ratio"),
            safe_div(pl.col("kmers_sample_exclade_count"), pl.col("kmers_sample_total_count")).alias("kmers_sample_excladeVStotal_ratio"),

            safe_div(pl.col("kmers_sample_root_count"), pl.col("kmers_sample_exclade_count")).alias("kmers_sample_rootVSexclade_ratio"),
            safe_div(pl.col("kmers_sample_lineage_count"), pl.col("kmers_sample_exclade_count")).alias("kmers_sample_lineageVSexclade_ratio"),
        ])
    )

    # 5. Calculate Weighted Distance/Depth Stats
    click.secho("Phase 3/4: Analyzing taxonomic distance distribution per sample...", fg="cyan", err=True)
    
    dist_base_lf = (
        labeled_lf
        .filter((~pl.col("is_in_clade")) & (pl.col("kmer_tax_id") > 0))
    )
    
    unique_pairs = dist_base_lf.select(["t_id", "kmer_tax_id"]).unique().collect()
    root_ids = unique_pairs["t_id"].to_numpy()
    kmer_ids = unique_pairs["kmer_tax_id"].to_numpy()
    
    lca_ids = tree.get_lca_batch(root_ids, kmer_ids)
    lca_depths = tree.depths[tree._get_indices(lca_ids)]
    
    pair_metrics_df = unique_pairs.with_columns([
        pl.Series("lca_depth", lca_depths.astype(np.uint8)),
        pl.Series("lca_id", lca_ids.astype(np.uint32))
    ])

    dist_lf = dist_base_lf.join(pair_metrics_df.lazy(), on=["t_id", "kmer_tax_id"]).with_columns([
        ((pl.col("root_depth") - pl.col("lca_depth")) + (pl.col("kmer_depth") - pl.col("lca_depth"))).alias("distance"),
        safe_div(pl.col("lca_depth"), (pl.col("root_depth").cast(pl.Int32) - 1)).alias("relative_lca_depth"),
        safe_div(pl.col("kmer_depth"), (pl.col("root_depth").cast(pl.Int32) - 1)).alias("relative_k_depth")
    ])
    
    misclassified_dist_lf = dist_lf.filter(~pl.col("is_in_lineage"))
    
    dist_stats = calculate_weighted_stats(misclassified_dist_lf, "distance", "kmer_count", ["sample_id", "t_id"], "kmers_sample_misclassified_dist")
    depth_stats = calculate_weighted_stats(misclassified_dist_lf, "kmer_depth", "kmer_count", ["sample_id", "t_id"], "kmers_sample_misclassified_depth")
    lca_stats = calculate_weighted_stats(misclassified_dist_lf, "relative_lca_depth", "kmer_count", ["sample_id", "t_id"], "kmers_sample_misclassified_relative_lca_depth")
    
    lineage_dist_lf = dist_lf.filter(pl.col("is_in_lineage"))
    lineage_stats = calculate_weighted_stats(lineage_dist_lf, "relative_k_depth", "kmer_count", ["sample_id", "t_id"], "kmers_sample_lineage_relative_depth")

    # 6. Top Hits
    click.secho("Phase 4/4: Identifying top 3 taxa for out-of-clade k-mer hits per sample...", fg="cyan", err=True)
    
    exclade_pooled = dist_lf.collect()
    misclassified_pooled = exclade_pooled.filter(~pl.col("is_in_lineage"))
    
    top_hits = (
        misclassified_pooled
        .group_by(["sample_id", "t_id"])
        .agg([
            pl.col("kmer_tax_id").sort_by(["kmer_count", "kmer_tax_id"], descending=[True, False]).head(3).alias("kmers_sample_misclassified_top3_taxids"),
            (pl.col("kmer_count").sort_by(["kmer_count", "kmer_tax_id"], descending=[True, False]).head(3) / pl.col("kmer_count").sum()).alias("kmers_sample_misclassified_top3_shares")
        ])
    ).join(
        exclade_pooled
        .group_by(["sample_id", "t_id"])
        .agg([
            pl.col("kmer_tax_id").sort_by(["kmer_count", "kmer_tax_id"], descending=[True, False]).head(3).alias("kmers_sample_exclade_top3_taxids"),
            (pl.col("kmer_count").sort_by(["kmer_count", "kmer_tax_id"], descending=[True, False]).head(3) / pl.col("kmer_count").sum()).alias("kmers_sample_exclade_top3_shares")
        ]), on=["sample_id", "t_id"], how="full", coalesce=True
    )
    
    all_top_ids = top_hits["kmers_sample_misclassified_top3_taxids"].explode().drop_nulls().unique().to_list()
    exclade_ids = top_hits["kmers_sample_exclade_top3_taxids"].explode().drop_nulls().unique().to_list()
    all_top_ids = list(set(all_top_ids + exclade_ids))
    
    if all_top_ids:
        names_df = tree.annotate(list(set(all_top_ids))).select(["t_id", "t_scientific_name"])
        names_map = dict(zip(names_df["t_id"].to_list(), names_df["t_scientific_name"].to_list()))
    else:
        names_map = {}
    
    top_hits = top_hits.with_columns([
        pl.col("kmers_sample_misclassified_top3_taxids").map_elements(
            lambda ids: [names_map.get(tid, "Unknown") for tid in ids] if ids is not None else None,
            return_dtype=pl.List(pl.String)
        ).alias("kmers_sample_misclassified_top3_names"),
        pl.col("kmers_sample_exclade_top3_taxids").map_elements(
            lambda ids: [names_map.get(tid, "Unknown") for tid in ids] if ids is not None else None,
            return_dtype=pl.List(pl.String)
        ).alias("kmers_sample_exclade_top3_names")
    ])

    # 7. Final Join & Save
    click.secho(f"Merging all features into {output_file.name}...", fg="cyan", err=True)
    
    # We maintain a strictly ordered schema for readability
    counts_cols = [
        "kmers_sample_total_count",
        "kmers_sample_classified_count",
        "kmers_sample_unclassified_count",
        "kmers_sample_clade_count",
        "kmers_sample_exclade_count",
        "kmers_sample_lineage_count",
        "kmers_sample_root_count"
    ]
    ratio_cols = [
        "kmers_sample_cladeVSclassified_ratio",
        "kmers_sample_lineageVSclassified_ratio",
        "kmers_sample_misclassifiedVSclassified_ratio",
        "kmers_sample_rootVSclassified_ratio",
        "kmers_sample_cladeVStotal_ratio",
        "kmers_sample_classifiedVStotal_ratio",
        "kmers_sample_lineageVStotal_ratio",
        "kmers_sample_rootVStotal_ratio",
        "kmers_sample_misclassifiedVStotal_ratio",
        "kmers_sample_excladeVStotal_ratio",
        "kmers_sample_supportingVStotal_ratio",
        "kmers_sample_rootVSexclade_ratio",
        "kmers_sample_lineageVSexclade_ratio"
    ]
    
    final_df = (
        counts_lf.collect()
        .join(dist_stats.collect(), on=["sample_id", "t_id"], how="left")
        .join(depth_stats.collect(), on=["sample_id", "t_id"], how="left")
        .join(lca_stats.collect(), on=["sample_id", "t_id"], how="left")
        .join(lineage_stats.collect(), on=["sample_id", "t_id"], how="left")
        .join(top_hits, on=["sample_id", "t_id"], how="left")
    )

    # Get structural stat columns (ordered by calculate_weighted_stats)
    struct_cols = []
    for suffix in ["kmers_sample_misclassified_dist", "kmers_sample_misclassified_depth", "kmers_sample_misclassified_relative_lca_depth", "kmers_sample_lineage_relative_depth"]:
        struct_cols.extend([f"{suffix}_{m}" for f in [suffix] for m in ["mean", "median", "stdev", "cv", "p05", "p95"]])

    top_cols = [c for c in top_hits.columns if c not in ["sample_id", "t_id"]]

    # Standardize column order
    final_df = final_df.select([
        "sample_id", "t_id",
        *counts_cols,
        *ratio_cols,
        *struct_cols,
        *top_cols
    ]).sort(["sample_id", "t_id"])

    final_df.write_parquet(output_file, compression="zstd")

    first_file = list(Path(input_pattern).parent.glob(Path(input_pattern).name))[0]
    meta_src = read_companion_metadata(first_file) or {}

    meta = get_standard_metadata(
        file_type="READ_LEN_FEATURE_LONG_PARQUET", # Re-using this general structure
        source_path=Path(input_pattern).parent,
        compression="zstd",
        sorting="sample_id, t_id",
        data_standard=meta_src.get("data_standard", "MIXED")
    )
    save_companion_metadata(output_file, meta)

    return {
        "records_processed": final_df.height,
        "unique_taxa": final_df["t_id"].n_unique(),
        "output_file": str(output_file)
    }

def produce_feature_kmer_stability_logic(
    input_parquet: Path,
    output_file: Path,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates the inter-sample Stability feature generation engine.

    This engine calculates the variance and consistency of the 13 classification
    quality ratios across all samples to provide a robust signal to machine learning
    models distinguishing between stable biological signal and erratic noise.

    Args:
        input_parquet : Path to the long-format .parquet file generated by kmer-sample.
        output_file   : Path to save the resulting feature table (CSV, TSV, or Parquet).
        cores         : Number of threads to dedicate to Polars.
        overwrite     : Whether to overwrite an existing output file.

    Returns:
        A dictionary containing processing statistics.
    """
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace.")

    check_write_permission(output_file)

    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    click.secho(f"Scanning sample data from '{input_parquet.name}'...", fg="cyan", err=True)
    lf = pl.scan_parquet(input_parquet)
    
    # Identify total number of unique samples to calculate correct project-wide occupancy
    total_samples = lf.select("sample_id").unique().collect().height
    click.secho(f"Found {total_samples} unique samples in dataset.", fg="cyan", err=True)

    click.secho("Calculating stability features across samples...", fg="cyan", err=True)

    TARGET_RATIOS = [
        "kmers_sample_cladeVSclassified_ratio",
        "kmers_sample_lineageVSclassified_ratio",
        "kmers_sample_misclassifiedVSclassified_ratio",
        "kmers_sample_rootVSclassified_ratio",
        "kmers_sample_cladeVStotal_ratio",
        "kmers_sample_classifiedVStotal_ratio",
        "kmers_sample_lineageVStotal_ratio",
        "kmers_sample_rootVStotal_ratio",
        "kmers_sample_misclassifiedVStotal_ratio",
        "kmers_sample_excladeVStotal_ratio",
        "kmers_sample_supportingVStotal_ratio",
        "kmers_sample_rootVSexclade_ratio",
        "kmers_sample_lineageVSexclade_ratio"
    ]

    base_agg = [
        pl.len().alias("_taxon_sample_count"),
        (pl.len() / total_samples).alias("kmers_stability_occupancy_ratio")
    ]

    for ratio in TARGET_RATIOS:
        base_name = ratio.replace("kmers_sample_", "kmers_stability_")
        base_agg.extend([
            pl.col(ratio).mean().alias(f"{base_name}_mean"),
            pl.col(ratio).median().alias(f"{base_name}_median"),
            
            # Polars std() correctly ignores nulls and yields null if n<=1
            pl.col(ratio).std().alias(f"{base_name}_stdev"),
            
            pl.col(ratio).quantile(0.05).alias(f"{base_name}_p05"),
            pl.col(ratio).quantile(0.95).alias(f"{base_name}_p95")
        ])

    agg_lf = lf.group_by("t_id").agg(base_agg)

    # Calculate derived stats: CV
    # We add CV back as a first-class feature for all ratios
    final_exprs = []
    for ratio in TARGET_RATIOS:
        base_name = ratio.replace("kmers_sample_", "kmers_stability_")
        final_exprs.append(
            pl.when(pl.col(f"{base_name}_mean") != 0)
              .then(pl.col(f"{base_name}_stdev") / pl.col(f"{base_name}_mean"))
              .otherwise(None)
              .alias(f"{base_name}_cv")
        )

    final_lf = agg_lf.with_columns(final_exprs).drop([
        c for c in agg_lf.collect_schema().names() if c.startswith("_") and c != "_taxon_sample_count"
    ]).drop("_taxon_sample_count")

    # 7. Reorder columns to group by metric
    ordered_cols = ["t_id", "kmers_stability_occupancy_ratio"]
    for ratio in TARGET_RATIOS:
        base_name = ratio.replace("kmers_sample_", "kmers_stability_")
        ordered_cols.extend([
            f"{base_name}_mean",
            f"{base_name}_median",
            f"{base_name}_stdev",
            f"{base_name}_cv",
            f"{base_name}_p05",
            f"{base_name}_p95"
        ])

    final_lf = final_lf.select(ordered_cols)

    click.secho(f"Merging and writing output to {output_file.name}...", fg="cyan", err=True)
    df = final_lf.collect().sort("t_id")
    
    ext = output_file.suffix.lower()
    if ext == ".parquet":
        df.write_parquet(output_file, compression="zstd")
    elif ext == ".tsv":
        df.write_csv(output_file, separator="\t", quote_style="non_numeric")
    else:
        df.write_csv(output_file, quote_style="non_numeric")

    meta_src = read_companion_metadata(input_parquet) or {}
    meta = get_standard_metadata(
        file_type="FEATURE_TABLE",
        source_path=input_parquet,
        compression="zstd" if ext == ".parquet" else "None",
        sorting="t_id",
        data_standard=meta_src.get("data_standard", "MIXED")
    )
    save_companion_metadata(output_file, meta)

    return {
        "taxa_processed": df.height,
        "output_file": str(output_file)
    }

def collect_feature_chunks_logic(
    input_pattern: str,
    output_file: Path,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Concatenates multiple intermediate feature chunks (Parquet) into a single file.

    This tool is designed for "Divide and Conquer" workflows where individual 
    samples are processed separately to stay within memory limits. It performs 
    a lazy concatenation and ensures that the final file is correctly sorted 
    and assigned valid provenance metadata.

    Args:
        input_pattern : Glob pattern for the intermediate .parquet chunks.
        output_file   : Path to save the final concatenated Parquet file.
        cores         : Number of threads to dedicate to Polars.
        overwrite     : Whether to overwrite an existing output file.

    Returns:
        A dictionary containing processing statistics.
    """
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace.")

    check_write_permission(output_file)

    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    # 1. Identify input files
    input_files = sorted(list(Path(input_pattern).parent.glob(Path(input_pattern).name)))
    if not input_files:
        raise FileNotFoundError(f"No files found matching pattern: '{input_pattern}'")

    click.secho(f"Concatenating {len(input_files)} feature chunks into {output_file.name}...", fg="cyan", err=True)

    # 2. Verify Schemas & Metadata
    # We check the first file to establish the 'master' metadata profile
    master_meta = read_companion_metadata(input_files[0]) or {}
    
    # Lazy scan all files
    lfs = [pl.scan_parquet(f) for f in input_files]
    
    # 3. Concatenate and Sort
    # Polars concatenation is extremely efficient (often O(1) metadata join)
    concatenated_lf = pl.concat(lfs, how="vertical")
    
    # We enforce consistent sorting for the final long-format dataset
    final_df = concatenated_lf.sort(["sample_id", "t_id"]).collect()

    # 4. Save and Metadata
    final_df.write_parquet(output_file, compression="zstd")

    meta = get_standard_metadata(
        file_type=master_meta.get("file_type", "FEATURE_TABLE"),
        source_path=Path(input_pattern).parent,
        compression="zstd",
        sorting="sample_id, t_id",
        data_standard=master_meta.get("data_standard", "MIXED")
    )
    save_companion_metadata(output_file, meta)

    return {
        "chunks_merged": len(input_files),
        "total_records": final_df.height,
        "output_file": str(output_file)
    }


def produce_feature_abundance_logic(
    input_table: Path,
    output_file: Path,
    inspect_file: Optional[Path] = None,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Calculates global abundance features (mean, median, stdev, p05, p95, CV) for every taxon.

    This engine consumes a wide-format abundance table (e.g., CLR-transformed or 
    proportions) and calculates row-wise statistical summaries across all samples.
    If a Kraken inspect file is provided, it also calculates minimizer-normalized 
    abundance features (abundance / clade_minimizers).

    Args:
        input_table  : Path to the wide-format abundance table.
        output_file  : Path to save the resulting feature table (.csv, .tsv, .parquet).
        inspect_file : (Optional) Path to Kraken inspect CSV file.
        cores        : Number of threads to dedicate to Polars.
        overwrite    : Whether to overwrite an existing output file.

    Returns:
        A dictionary containing processing statistics.
    """
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace.")

    check_write_permission(output_file)

    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    click.secho(f"Loading abundance table from '{input_table.name}'...", fg="cyan", err=True)

    # 1. Load table
    ext_in = input_table.suffix.lower()
    if ext_in == ".parquet":
        lf = pl.scan_parquet(input_table)
    elif ext_in == ".tsv" or input_table.name.endswith(".tsv.gz"):
        lf = pl.scan_csv(input_table, separator="\t")
    else:
        lf = pl.scan_csv(input_table)

    # 2. Identify sample columns
    schema = lf.collect_schema()
    sample_cols = [c for c in schema.names() if c != "t_id"]

    if not sample_cols:
        raise ValueError("The input table must contain at least one sample column in addition to 't_id'.")

    # 3. Handle Minimizers (Optional)
    if inspect_file:
        click.secho(f"Integrating minimizer normalization from '{inspect_file.name}'...", fg="cyan", err=True)
        inspect_lf = pl.scan_csv(inspect_file).select([
            pl.col("tax_id").alias("t_id"),
            pl.col("clade_minimizers").cast(pl.Float64)
        ])
        # Join inspect data to the main table
        lf = lf.join(inspect_lf, on="t_id", how="left")

    click.secho(f"Calculating abundance statistics across {len(sample_cols)} samples...", fg="cyan", err=True)

    # 4. Calculate statistics
    unpivot_cols = ["t_id"]
    if inspect_file:
        unpivot_cols.append("clade_minimizers")

    long_lf = lf.unpivot(index=unpivot_cols, variable_name="sample_id", value_name="abundance")

    # Base expressions
    agg_exprs = [
        pl.col("abundance").mean().alias("abund_global_mean"),
        pl.col("abundance").median().alias("abund_global_median"),
        pl.col("abundance").std().alias("abund_global_stdev"),
        pl.col("abundance").quantile(0.05, interpolation="nearest").alias("abund_global_p05"),
        pl.col("abundance").quantile(0.95, interpolation="nearest").alias("abund_global_p95"),
    ]

    # Add minimizer-normalized expressions if available
    if inspect_file:
        # Avoid division by zero/null for minimizers
        normalized_expr = pl.col("abundance") / pl.col("clade_minimizers")
        agg_exprs.extend([
            normalized_expr.mean().alias("abund_global_meanVSmm_ratio"),
            normalized_expr.median().alias("abund_global_medianVSmm_ratio")
        ])

    stats_lf = long_lf.group_by("t_id").agg(agg_exprs)

    # Calculate CV (must be done after mean/stdev are calculated)
    stats_lf = stats_lf.with_columns(
        (pl.col("abund_global_stdev") / pl.col("abund_global_mean")).alias("abund_global_cv")
    ).sort("t_id")

    # 5. Reorder columns
    ordered_cols = [
        "t_id", 
        "abund_global_mean", 
        "abund_global_median", 
        "abund_global_stdev", 
        "abund_global_cv", 
        "abund_global_p05", 
        "abund_global_p95"
    ]
    if inspect_file:
        ordered_cols.extend(["abund_global_meanVSmm_ratio", "abund_global_medianVSmm_ratio"])
    
    df = stats_lf.select(ordered_cols).collect()

    # 6. Save
    click.secho(f"Saving abundance features to {output_file.name}...", fg="cyan", err=True)
    ext_out = output_file.suffix.lower()
    if ext_out == ".parquet":
        df.write_parquet(output_file, compression="zstd")
    elif ext_out == ".tsv":
        df.write_csv(output_file, separator="\t", quote_style="non_numeric")
    else:
        df.write_csv(output_file, quote_style="non_numeric")

    # 6. Metadata
    out_meta = get_standard_metadata(
        file_type="FEATURE_TABLE",
        source_path=input_table,
        compression="zstd" if ext_out == ".parquet" else "None",
        sorting="t_id",
        data_standard="MIXED"
    )
    save_companion_metadata(output_file, out_meta)

    return {
        "taxa_processed": df.height,
        "sample_count": len(sample_cols),
        "output_file": str(output_file)
    }
