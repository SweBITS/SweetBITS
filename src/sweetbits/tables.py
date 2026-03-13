"""
sweetbits.tables
Logic for generating abundance matrices and calculating canonical remainders.
"""

import polars as pl
import logging
import click
from pathlib import Path
from typing import Optional, Dict, Any
from joltax import JolTree
from sweetbits.utils import load_sample_id_list, FILTERED_TID, UNCLASSIFIED_TID, check_write_permission
from sweetbits.metadata import get_standard_metadata, save_companion_metadata, validate_sweetbits_file
from sweetbits.taxmath import calc_clade_sum
from sweetbits.canonical import calculate_canonical_remainders
from sweetbits.audit import print_audit_report, _aggregate_reads_by_rank

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
    keep_filtered: bool = False,
    cores: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False
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
        keep_filtered     : If True (taxon/canonical modes only), retains filtered reads in a 
                            synthetic 'Filtered classified' bin to preserve the global read total
                            for accurate relative abundance calculations.
        cores             : Number of CPU cores to use for Polars operations.
        overwrite         : Whether to overwrite the output file if it exists.
        dry_run           : If True, prints the audit report and returns without saving.

    Returns:
        A dictionary containing processing statistics:
        - 'data_standard'  : Detected standard (SWEBITS/GENERIC).
        - 'active_samples' : Number of samples included in the output.
        - 'rows_output'    : Number of taxa in the final table.
        - 'output_file'    : Path to the saved result.

    Raises:
        ValueError        : If required parameters are missing for the selected mode.
        FileNotFoundError : If the input file does not exist.
        FileExistsError   : If output_file exists and overwrite is False.
    """
    click.secho("Initiating table generation...", fg="cyan", err=True)

    if not overwrite and output_file.exists():
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace it.")

    # 0. Early Validations
    if not dry_run:
        check_write_permission(output_file)
        
    if cores:
        import os
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    # 1. Validate Parquet and Read Metadata via JSON Companion
    required_cols = ["sample_id", "t_id", "taxon_reads"]
    metadata = validate_sweetbits_file(input_parquet, expected_type="REPORT_PARQUET", required_columns=required_cols)
    data_standard = metadata.get("data_standard", "GENERIC")

    # Lazy scan and early normalization
    lf = pl.scan_parquet(input_parquet)
    
    if data_standard == "SWEBITS":
        # Consolidate SWEBITS samples into standard 'sample_id' format immediately
        # so downstream logic is completely agnostic to the data standard.
        lf = lf.with_columns(
            (pl.col("year").cast(pl.String) + "_" + pl.col("week").cast(pl.String).str.pad_start(2, "0")).alias("period")
        ).drop("sample_id").rename({"period": "sample_id"})
        
    # Isolate required columns and collect into eager memory ONCE. 
    # This prevents multiple disk I/O passes for validation, true_totals, and math.
    input_df = lf.select(["t_id", "sample_id", "taxon_reads"]).collect()

    # 2. Sample Filtering and Validation
    all_ids = set(input_df["sample_id"].unique().to_list())
    excluded_ids = []
    phantom_ids = []

    if exclude_samples:
        click.secho("Applying sample exclusions...", fg="cyan", err=True)
        excluded_ids = load_sample_id_list(exclude_samples)
        phantom_ids = [eid for eid in excluded_ids if eid not in all_ids]
        
        if phantom_ids:
            click.secho(
                f"Warning: {len(phantom_ids)} sample IDs in exclusion file were not found in the dataset. "
                "Please check for typos.", fg="yellow", err=True
            )
        input_df = input_df.filter(~pl.col("sample_id").is_in(excluded_ids))
        
    active_samples = input_df.select("sample_id").n_unique()
        
    if min_observed > (active_samples / 2) and active_samples > 0:
        click.secho(
            f"Warning: --min-observed ({min_observed}) is more than 50% of active samples ({active_samples}).", 
            fg="yellow", err=True
        )

    # 3. Baseline Computation (Global Totals)
    true_totals = {}
    if keep_filtered:
        click.secho("Calculating baseline totals for keep-filtered logic...", fg="cyan", err=True)
        totals_df = input_df.group_by("sample_id").agg(pl.col("taxon_reads").sum().alias("total_reads"))
        true_totals = dict(zip(totals_df["sample_id"].to_list(), totals_df["total_reads"].to_list()))

    # 4. Taxonomic Filtering (JolTax Integration)
    if not taxonomy_dir:
        raise ValueError("Taxonomy directory is required for table generation.")
        
    click.secho("Loading JolTax taxonomy tree...", fg="cyan", err=True)
    tree = JolTree.load(str(taxonomy_dir))
        
    if clade_filter is not None:
        click.secho(f"Applying clade filter for TaxID {clade_filter}...", fg="cyan", err=True)
        clade_taxids = tree.get_clade(clade_filter)
        input_df = input_df.filter(pl.col("t_id").is_in(clade_taxids))
        
    if not keep_unclassified and mode != "canonical":
        click.secho("Filtering out unclassified reads...", fg="cyan", err=True)
        input_df = input_df.filter(pl.col("t_id") != 0)
        
    # 5. Dynamic Clade Math & Filtering
    click.secho("Applying dynamic recursive filtering and calculating clades...", fg="cyan", err=True)
        
    # Calculate baseline for audit report (no filters)
    baseline_df, _ = calc_clade_sum(
        input_df, tree, min_reads=0, min_observed=0, keep_filtered=False
    )
    
    # Calculate filtered for actual output
    filtered_df, synthetic_bin = calc_clade_sum(
        input_df, tree, min_reads=min_reads, min_observed=min_observed, keep_filtered=keep_filtered
    )
    
    # Prune rows where clade_reads is 0 (except Unclassified)
    baseline_df = baseline_df.filter((pl.col("clade_reads") > 0) | (pl.col("t_id") == UNCLASSIFIED_TID))
    filtered_df = filtered_df.filter((pl.col("clade_reads") > 0) | (pl.col("t_id") == UNCLASSIFIED_TID))
    
    if clade_filter is not None:
        clade_taxids = tree.get_clade(clade_filter)
        baseline_df = baseline_df.filter(pl.col("t_id").is_in(clade_taxids))
        filtered_df = filtered_df.filter(pl.col("t_id").is_in(clade_taxids))
    
    # 6. Mode Extraction and Matrix Generation
    click.secho(f"Extracting '{mode}' metrics...", fg="cyan", err=True)
    if mode == "taxon":
        pivot_df = filtered_df.select(["t_id", "sample_id", "taxon_reads"])
        pivot_col = "taxon_reads"
    elif mode == "clade":
        pivot_df = filtered_df.select(["t_id", "sample_id", "clade_reads"])
        pivot_col = "clade_reads"
    elif mode == "canonical":
        click.secho("Calculating canonical remainders (this may take a moment)...", fg="cyan", err=True)
        pivot_df = calculate_canonical_remainders(
            filtered_df, 
            tree, 
            keep_unclassified=keep_unclassified,
            clade_filter=clade_filter
        )
        pivot_col = "val"

    click.secho("Pivoting data to wide format matrix...", fg="cyan", err=True)
    table = pivot_df.pivot(
        values=pivot_col,
        index="t_id",
        on="sample_id",
        aggregate_function="sum"
    ).fill_null(0).sort("t_id")
    
    # 6.5 Capture Baseline Metrics for Audit Report
    baseline_taxa_count = baseline_df.select("t_id").n_unique()
    base_tids = baseline_df.select("t_id").unique()["t_id"].to_list()
    
    sample_cols = [c for c in table.columns if c != "t_id"]
    
    baseline_reads = 0
    retained_reads = 0
    if not proportions and sample_cols:
        baseline_reads = baseline_df.select(pl.col("taxon_reads").sum()).item()
        retained_reads = filtered_df.select(pl.col("taxon_reads").sum()).item()
        
    # 7. Composition Preservation
    produced_synthetic = False
    if keep_filtered and sample_cols:
        click.secho("Applying keep-filtered logic to preserve mass balance...", fg="cyan", err=True)
        filtered_row = {"t_id": FILTERED_TID}
        has_filtered = False
        
        surviving_totals = filtered_df.group_by("sample_id").agg(pl.col("taxon_reads").sum().alias("survived"))
        surviving_map = dict(zip(surviving_totals["sample_id"].to_list(), surviving_totals["survived"].to_list()))

        for c in sample_cols:
            survived = surviving_map.get(c, 0)
            original_total = true_totals.get(c, survived)
            diff = max(0, original_total - survived)
            filtered_row[c] = diff
            if diff > 0:
                has_filtered = True
                
        if has_filtered:
            filtered_df_table = pl.DataFrame([filtered_row], schema=table.schema)
            table = pl.concat([table, filtered_df_table])
            produced_synthetic = True
            
    # 8. Proportion Calculation
    if proportions and sample_cols:
        click.secho("Converting counts to relative proportions...", fg="cyan", err=True)
        if mode in ["taxon", "canonical"]:
            exprs = [(pl.col(c) / pl.col(c).sum()).alias(c) for c in sample_cols]
            table = table.with_columns(exprs)
        elif mode == "clade":
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
            
    # 9. Audit Report Calculation and Printing
    base_clade_dict = None
    retained_clade_dict = None
    base_taxon_dict = None
    retained_taxon_dict = None
    if tree and sample_cols and not proportions:
        base_clade_dict = _aggregate_reads_by_rank(baseline_df, tree, "clade_reads")
        retained_clade_dict = _aggregate_reads_by_rank(filtered_df, tree, "clade_reads")
        base_taxon_dict = _aggregate_reads_by_rank(baseline_df, tree, "taxon_reads")
        retained_taxon_dict = _aggregate_reads_by_rank(filtered_df, tree, "taxon_reads")
    
    print_audit_report(
        dry_run=dry_run,
        input_name=input_parquet.name,
        total_samples=len(all_ids),
        actual_excluded=len(excluded_ids) - len(phantom_ids) if exclude_samples else 0,
        active_samples=active_samples,
        proportions=proportions,
        mode=mode,
        baseline_reads=baseline_reads,
        retained_reads=retained_reads,
        produced_synthetic=produced_synthetic,
        keep_filtered=keep_filtered,
        has_unclass=UNCLASSIFIED_TID in table["t_id"].to_list(),
        tree=tree,
        base_tids=base_tids,
        final_tids=table["t_id"].to_list(),
        baseline_taxa_count=baseline_taxa_count,
        final_taxa_count=table.height,
        num_sample_cols=len(sample_cols),
        base_clade_reads=base_clade_dict,
        retained_clade_reads=retained_clade_dict,
        base_taxon_reads=base_taxon_dict,
        retained_taxon_reads=retained_taxon_dict
    )
    
    if dry_run:
        click.secho("Dry-run complete. Exiting without saving.", fg="yellow", bold=True, err=True)
        return {
            "data_standard": data_standard,
            "active_samples": active_samples,
            "rows_output": table.height,
            "output_file": "DRY_RUN"
        }
        
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
