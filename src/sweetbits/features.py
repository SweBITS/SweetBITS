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
            ).alias("mm_pearson_coeff"),
            
            pl.when(n >= MIN_OBSERVATIONS_FOR_CORR).then(
                pl.when(has_variance).then(
                    pl.corr(x, y) * ((n - 2) / (1 - pl.corr(x, y).pow(2))).sqrt()
                ).otherwise(None)
            ).alias("_t_stat"),
            
            n.alias("mm_pearson_n"),

            pl.when(n >= MIN_OBSERVATIONS_FOR_CORR).then(
                pl.when(has_variance_f).then(pl.corr(x_f, y_f)).otherwise(None)
            ).alias("mm_filtered_coeff"),

            pl.when(n >= MIN_OBSERVATIONS_FOR_CORR).then(
                pl.when(has_variance_f).then(
                    pl.corr(x_f, y_f) * ((n_f - 2) / (1 - pl.corr(x_f, y_f).pow(2))).sqrt()
                ).otherwise(None)
            ).alias("_t_stat_f"),

            n_f.alias("mm_filtered_n"),

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
            
            pl.struct(["_t_stat_f", "mm_filtered_n"]).map_elements(
                lambda s: calculate_p_value(s["_t_stat_f"], s["mm_filtered_n"]),
                return_dtype=pl.Float64
            ).alias("mm_filtered_p")
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
        "valid_correlations": summary_df.filter(pl.col("mm_pearson_coeff").is_not_null()).height,
        "output_format": ext[1:].upper(),
        "long_format_saved": True if output_long_file else False
    }
