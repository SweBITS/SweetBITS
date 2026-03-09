"""
sweetbits.tables
Logic for generating abundance matrices and calculating canonical remainders.
"""

import polars as pl
import logging
import numpy as np
import click
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from joltax import JolTree
from joltax.constants import CANONICAL_RANKS
from sweetbits.utils import parse_sample_id, load_sample_id_list, FILTERED_TID
from sweetbits.metadata import get_standard_metadata, save_companion_metadata, read_companion_metadata, validate_sweetbits_file

from sweetbits.canonical import calculate_canonical_remainders

logger = logging.getLogger(__name__)

def generate_table_logic(
    input_parquet: Path,
    output_file: Path,
    mode: str = "clade",
    taxonomy_dir: Optional[Path] = None,
    exclude_samples: Optional[Path] = None,
    min_observed: int = 25,
    min_reads: int = 50,
    clade_filter: Optional[int] = None,
    keep_unclassified: bool = False,
    proportions: bool = False,
    keep_composition: bool = False,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Generates a wide-format abundance table from a merged REPORT_PARQUET file.

    This function automatically detects the 'data_standard' from the input file's
    metadata to decide how to format columns (YYYY_WW vs Sample ID). It supports
    filtering by clade, sample exclusion, quality thresholds, and composition
    preservation.

    Args:
        input_parquet     : Path to the merged REPORT_PARQUET file.
        output_file       : Path where the table will be saved (.csv, .tsv, or .parquet).
        mode              : Abundance calculation mode:
                            - 'taxon': Raw reads assigned directly to a TaxID.
                            - 'clade': Cumulative reads for a taxon and all descendants.
                            - 'canonical': Canonical remainders. Essentially taxon mode but where 
                                           reads between canonical ranks have been pushed up to 
                                           the nearest canonical ancestor (NCA). Ensures mass balance.
        taxonomy_dir      : Path to JolTax cache directory (required for 'canonical' or clade filter).
        exclude_samples   : Optional path to a text file containing Sample IDs to exclude.
        min_observed      : Minimum number of samples a taxon must appear in.
        min_reads         : Minimum maximum read count across all samples for a taxon.
        clade_filter      : Optional TaxID to restrict output to a specific clade.
        keep_unclassified : Whether to include TaxID 0 in the output.
        proportions       : If True, outputs relative proportions instead of raw read counts.
        keep_composition  : If True (taxon/canonical modes only), retains filtered reads in a 
                            synthetic 'Filtered classified' bin to preserve the global read total
                            for accurate relative abundance calculations.
        cores             : Number of CPU cores to use for Polars operations.
        overwrite         : Whether to overwrite the output file if it exists.

    Returns:
        A dictionary containing processing statistics:
        - 'data_standard'  : Detected standard (SWEBITS/GENERIC).
        - 'active_samples' : Number of samples included in the output.
        - 'rows_output'    : Number of taxa in the final table.
        - 'output_file'    : Path to the saved result.

    Raises:
        ValueError        : If required parameters are missing for the selected mode, or if 
                            keep_composition is used with an incompatible mode.
        FileNotFoundError : If the input file does not exist.
        FileExistsError   : If output_file exists and overwrite is False.
    """
    click.secho("Initiating table generation...", fg="cyan", err=True)

    if not overwrite and output_file.exists():
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace it.")

    if cores:
        import os
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    # 1. Validate Parquet and Read Metadata via JSON Companion
    required_cols = ["sample_id", "t_id", "clade_reads", "taxon_reads"]
    metadata = validate_sweetbits_file(input_parquet, expected_type="REPORT_PARQUET", required_columns=required_cols)
    data_standard = metadata.get("data_standard", "GENERIC")

    # Lazy scan - Polars natively handles the Categorical sample_id column written by collect
    lf = pl.scan_parquet(input_parquet)

    # 2. Sample Filtering and Validation
    # Execute ONE quick scan of just the sample IDs to power validation and math
    existing_samples_df = lf.select("sample_id").unique().collect()
    all_ids = set(existing_samples_df["sample_id"].to_list())

    if exclude_samples:
        click.secho("Applying sample exclusions...", fg="cyan", err=True)
        excluded_ids = load_sample_id_list(exclude_samples)
        phantom_ids = [eid for eid in excluded_ids if eid not in all_ids]
        
        if phantom_ids:
            click.secho(
                f"Warning: {len(phantom_ids)} sample IDs in exclusion file were not found in the dataset. "
                "Please check for typos.", fg="yellow", err=True
            )
        lf = lf.filter(~pl.col("sample_id").is_in(excluded_ids))
        
        # Calculate active samples mathematically without reading the file a second time
        active_samples = len(all_ids) - (len(excluded_ids) - len(phantom_ids))
    else:
        active_samples = len(all_ids)
        
    if min_observed > (active_samples / 2) and active_samples > 0:
        click.secho(
            f"Warning: --min-observed ({min_observed}) is more than 50% of active samples ({active_samples}).", 
            fg="yellow", err=True
        )

    # 3. Baseline Computation (Global Totals)
    # If the user wants to keep the composition (retain filtered reads as 'Filtered classified'),
    # we must calculate the true, unfiltered total reads for each sample BEFORE any taxonomic
    # filters drop rows from the lazyframe.
    true_totals = {}
    if keep_composition:
        click.secho("Calculating baseline totals for keep-composition logic...", fg="cyan", err=True)
        if mode == "clade":
            raise ValueError("--keep-composition is not mathematically valid for 'clade' mode due to read double-counting. Please use 'taxon' or 'canonical' mode.")
            
        tot_lf = lf
        if data_standard == "SWEBITS":
            tot_lf = tot_lf.with_columns(
                (pl.col("year").cast(pl.String) + "_" + pl.col("week").cast(pl.String).str.pad_start(2, "0")).alias("period")
            )
            pkey = "period"
        else:
            pkey = "sample_id"
            
        totals_df = tot_lf.group_by(pkey).agg(pl.col("taxon_reads").sum().alias("total_reads")).collect()
        true_totals = dict(zip(totals_df[pkey].to_list(), totals_df["total_reads"].to_list()))

    # 4. Taxonomic Filtering (JolTax Integration)
    # Load the taxonomy tree if required for the specified mode or clade filter.
    tree = None
    if mode == "canonical" or clade_filter is not None:
        click.secho("Loading JolTax taxonomy tree...", fg="cyan", err=True)
        if not taxonomy_dir:
            raise ValueError(f"Taxonomy directory is required for mode '{mode}' or clade filtering.")
        tree = JolTree.load(str(taxonomy_dir))
        
    if clade_filter is not None:
        click.secho(f"Applying clade filter for TaxID {clade_filter}...", fg="cyan", err=True)
        clade_taxids = tree.get_clade(clade_filter)
        lf = lf.filter(pl.col("t_id").is_in(clade_taxids))
        
    if not keep_unclassified and mode != "canonical":
        click.secho("Filtering out unclassified reads...", fg="cyan", err=True)
        lf = lf.filter(pl.col("t_id") != 0)
        
    # 5. Metric Aggregation
    # Collect the necessary columns and perform the required mathematical transformations
    # based on the requested abundance mode.
    click.secho(f"Aggregating metrics for '{mode}' mode...", fg="cyan", err=True)
    target_cols = ["t_id", "sample_id"]
    if data_standard == "SWEBITS":
        target_cols.extend(["year", "week"])
        
    if mode == "taxon":
        target_cols.append("taxon_reads")
        pivot_df = lf.select(target_cols).collect()
        pivot_col = "taxon_reads"
    elif mode == "clade":
        target_cols.append("clade_reads")
        pivot_df = lf.select(target_cols).collect()
        pivot_col = "clade_reads"
    elif mode == "canonical":
        click.secho("Calculating canonical remainders (this may take a moment)...", fg="cyan", err=True)
        # NCA math always requires clade_reads, audit needs taxon_reads
        input_cols = ["t_id", "sample_id", "clade_reads", "taxon_reads"]
        input_df = lf.select(input_cols).collect()
        
        # Calculate remainders via optimized NCA logic
        pivot_df = calculate_canonical_remainders(
            input_df, 
            tree, 
            keep_unclassified=keep_unclassified,
            clade_filter=clade_filter
        )
        
        # Re-attach temporal columns if SWEBITS standard
        if data_standard == "SWEBITS":
            sample_meta = lf.select(["sample_id", "year", "week"]).unique().collect()
            pivot_df = pivot_df.join(sample_meta, on="sample_id")
        pivot_col = "val"

    # 6. Matrix Generation (Pivoting)
    # Transform the long-format data into a wide-format matrix. 
    # For SWEBITS data, group by YYYY_WW to enforce the strict project 
    # constraint of one unique sample per week per site.
    click.secho("Pivoting data to wide format matrix...", fg="cyan", err=True)
    if data_standard == "SWEBITS":
        pivot_df = pivot_df.with_columns(
            (pl.col("year").cast(pl.String) + "_" + pl.col("week").cast(pl.String).str.pad_start(2, "0")).alias("period")
        )
        pivot_key = "period"
    else:
        pivot_key = "sample_id"
    
    table = pivot_df.pivot(
        values=pivot_col,
        index="t_id",
        on=pivot_key,
        aggregate_function="sum"
    ).fill_null(0).sort("t_id")
    
    # 7. Quality Control Filters
    # Apply minimum occupancy (--min-observed) and abundance (--min-reads) thresholds.
    sample_cols = [c for c in table.columns if c != "t_id"]
    if sample_cols:
        if min_observed > 0:
            click.secho(f"Applying minimum required observations filter (min_observed >= {min_observed})...", fg="cyan", err=True)
            # Idiomatic Polars: evaluate directly inside the filter context
            table = table.filter(
                pl.sum_horizontal([(pl.col(c) > 0) for c in sample_cols]) >= min_observed
            )
            
        if min_reads > 0:
            click.secho(f"Applying minimum read depth filter (min_reads >= {min_reads})...", fg="cyan", err=True)
            table = table.filter(
                pl.max_horizontal(sample_cols) >= min_reads
            )
            
    # 8. Composition Preservation
    # If the user requested --keep-composition, we compare the current column sums
    # against the pre-calculated baseline totals. Any missing reads are bundled
    # into a synthetic "Filtered classified" bin to preserve the global sample size.
    if keep_composition and sample_cols:
        click.secho("Applying keep-composition logic to preserve mass balance...", fg="cyan", err=True)
        filtered_row = {"t_id": FILTERED_TID}
        has_filtered = False
        for c in sample_cols:
            current_sum = table[c].sum()
            original_total = true_totals.get(c, current_sum)
            diff = original_total - current_sum
            if diff < 0: diff = 0
            filtered_row[c] = diff
            if diff > 0:
                has_filtered = True
                
        if has_filtered:
            filtered_df = pl.DataFrame([filtered_row], schema=table.schema)
            table = pl.concat([table, filtered_df])
            
    # 9. Proportion Calculation
    # Convert raw counts to relative proportions if requested.
    if proportions and sample_cols:
        click.secho("Converting counts to relative proportions...", fg="cyan", err=True)
        if mode in ["taxon", "canonical"]:
            # Mutually exclusive modes naturally sum to 1.0
            exprs = [(pl.col(c) / pl.col(c).sum()).alias(c) for c in sample_cols]
            table = table.with_columns(exprs)
        elif mode == "clade":
            # Clade mode is cumulative. We divide by the local maximum classified node
            # plus any unclassified reads to represent the true local proportion.
            exprs = []
            for c in sample_cols:
                max_class = table.filter(pl.col("t_id") != 0)[c].max()
                if max_class is None: max_class = 0
                unclass = table.filter(pl.col("t_id") == 0)[c].sum()
                if unclass is None: unclass = 0
                total = max_class + unclass
                if total == 0: total = 1
                exprs.append((pl.col(c) / total).alias(c))
            table = table.with_columns(exprs)
        
    # 10. Output Generation
    click.secho(f"Saving output to {output_file.name}...", fg="cyan", err=True)
    ext = output_file.suffix.lower()
    meta = get_standard_metadata("RAW_TABLE", source_path=input_parquet, sorting="t_id", data_standard=data_standard)
    meta["mode"] = mode
    
    if ext == ".parquet":
        table.write_parquet(output_file)
    elif ext == ".csv":
        table.write_csv(output_file)
    elif ext == ".tsv":
        table.write_csv(output_file, separator="\t")
    else:
        raise ValueError(f"Unsupported output format: {ext}")
        
    save_companion_metadata(output_file, meta)
    
    click.secho("Done!", fg="cyan", bold=True, err=True)
        
    return {
        "data_standard": data_standard,
        "active_samples": active_samples,
        "rows_output": table.height,
        "output_file": str(output_file)
    }
