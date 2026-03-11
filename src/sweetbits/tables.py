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
from sweetbits.utils import parse_sample_id, load_sample_id_list, FILTERED_TID, UNCLASSIFIED_TID, check_write_permission
from sweetbits.metadata import get_standard_metadata, save_companion_metadata, read_companion_metadata, validate_sweetbits_file
from sweetbits.taxmath import calc_clade_sum
from sweetbits.canonical import calculate_canonical_remainders

logger = logging.getLogger(__name__)

def _print_audit_report(
    dry_run: bool,
    input_name: str,
    total_samples: int,
    actual_excluded: int,
    active_samples: int,
    proportions: bool,
    mode: str,
    baseline_reads: int,
    retained_reads: int,
    produced_synthetic: bool,
    keep_filtered: bool,
    has_unclass: bool,
    tree: Optional[JolTree],
    base_tids: List[int],
    final_tids: List[int],
    baseline_taxa_count: int,
    final_taxa_count: int,
    num_sample_cols: int,
    base_clade_reads: Optional[Dict[str, float]] = None,
    retained_clade_reads: Optional[Dict[str, float]] = None,
    base_taxon_reads: Optional[Dict[str, float]] = None,
    retained_taxon_reads: Optional[Dict[str, float]] = None
):
    """Prints the audit report for table generation."""
    click.secho("\n" + "="*80, fg="bright_black", err=True)
    header_text = "SweetBITS Table Audit (--dry-run)" if dry_run else "SweetBITS Table Audit"
    # Centering logic: (80 - len(text)) // 2
    padding = (80 - len(header_text)) // 2
    click.secho(" " * padding + header_text, fg="yellow", bold=True, err=True)
    click.secho("="*80 + "\n", fg="bright_black", err=True)
    
    click.secho("[ 1 ] Data & Sample Overview", fg="cyan", bold=True, err=True)
    click.secho("-" * 80, fg="bright_black", err=True)
    click.secho(f"Input Parquet         : {input_name}", err=True)
    click.secho(f"Total Samples in Data : {total_samples}", err=True)
    
    click.secho(f"Samples Excluded      : {actual_excluded}", err=True)
    samp_pct = (active_samples / total_samples * 100) if total_samples else 0
    click.secho(f"Samples Kept          : {active_samples} ({samp_pct:.2f}%)\n", err=True)

    if not proportions:
        click.secho("[ 2 ] Read Retention", fg="cyan", bold=True, err=True)
        click.secho("-" * 80, fg="bright_black", err=True)
        
        click.secho(f"Total Reads (Base)    : {baseline_reads:,}", err=True)
        filtered_out = baseline_reads - retained_reads
        click.secho(f"Filtered Out          : {filtered_out:,}", err=True)
        read_pct = (retained_reads / baseline_reads * 100) if baseline_reads > 0 else 0
        click.secho(f"Reads Retained        : {retained_reads:,} ({read_pct:.2f}%)", err=True)
        
        comp_status = "YES (Filtered reads retained in synthetic bin)" if produced_synthetic else "NO"
        
        if keep_filtered and not has_unclass:
             comp_status = "PARTIAL (Filtered reads kept, but Unclassified reads missing)"
        
        click.secho(f"Composition Intact    : {comp_status}", err=True)
        click.secho(f"Unclassified Kept     : {'YES' if has_unclass else 'NO'}\n", err=True)

    click.secho("[ 3 ] Taxonomic Retention", fg="cyan", bold=True, err=True)
    click.secho("-" * 80, fg="bright_black", err=True)
    
    if tree:
        # Calculate rank breakdown for original vs retained
        import numpy as np
        
        def count_ranks(tids_list):
            counts = {}
            t_arr = np.array(tids_list, dtype=np.uint32)
            valid_idx = tree._get_indices(t_arr)
            valid_idx = valid_idx[valid_idx != -1]
            if len(valid_idx) > 0:
                ranks = tree.ranks[valid_idx]
                for r in ranks:
                    r_name = tree.rank_names[r]
                    counts[r_name] = counts.get(r_name, 0) + 1
            return counts
            
        base_counts = count_ranks(base_tids)
        final_counts = count_ranks(final_tids)
        
        # Deduplicate while preserving order (Top Rank followed by standard Canonical Ranks)
        display_ranks = []
        for r in [tree.top_rank] + CANONICAL_RANKS:
            if r not in display_ranks:
                display_ranks.append(r)
        
        click.secho(f"{'Rank':<16} {'Original Count':<18} {'Retained Count':<18} {'Retention %':<12}", bold=True, err=True)
        click.secho("-" * 80, fg="bright_black", err=True)
        
        # Add Total Classified (distinct TaxIDs in tree)
        total_taxa_pct = (final_taxa_count / baseline_taxa_count * 100) if baseline_taxa_count > 0 else 0
        click.secho(f"{'Classified':<16} {baseline_taxa_count:<18} {final_taxa_count:<18} {total_taxa_pct:.2f}%", bold=True, err=True)
        
        for rank in display_ranks:
            if rank in base_counts:
                o_c = base_counts[rank]
                r_c = final_counts.get(rank, 0)
                pct = (r_c / o_c * 100) if o_c > 0 else 0
                click.secho(f"{rank.capitalize():<16} {o_c:<18} {r_c:<18} {pct:.2f}%", err=True)
        click.secho("-" * 80, fg="bright_black", err=True)
    
    # Space between sections
    click.echo("", err=True)

    if base_clade_reads is not None and retained_clade_reads is not None:
        click.secho("[ 4 ] Read Retention at CLADE Level by Canonical Rank", fg="cyan", bold=True, err=True)
        click.secho("-" * 80, fg="bright_black", err=True)
        click.secho(f"{'Rank':<16} {'Original Reads':<18} {'Retained Reads':<18} {'Retention %':<12}", bold=True, err=True)
        click.secho("-" * 80, fg="bright_black", err=True)
        
        for rank in display_ranks:
            if rank in base_clade_reads:
                o_r = base_clade_reads[rank]
                r_r = retained_clade_reads.get(rank, 0.0)
                pct = (r_r / o_r * 100) if o_r > 0 else 0
                click.secho(f"{rank.capitalize():<16} {int(o_r):<18,} {int(r_r):<18,} {pct:.2f}%", err=True)
        click.secho("-" * 80, fg="bright_black", err=True)
        click.secho(" This table sums cumulative clade counts at each rank. Reads are multi-counted", fg="bright_black", italic=True, err=True)
        click.secho(" across ranks. Non-monotonicity may occur if lineages skip standard ranks.", fg="bright_black", italic=True, err=True)
        click.echo("", err=True)

    if base_taxon_reads is not None and retained_taxon_reads is not None:
        click.secho("[ 5 ] Read Retention at TAXON Level by Canonical Rank", fg="cyan", bold=True, err=True)
        click.secho("-" * 80, fg="bright_black", err=True)
        click.secho(f"{'Rank':<16} {'Original Reads':<18} {'Retained Reads':<18} {'Retention %':<12}", bold=True, err=True)
        click.secho("-" * 80, fg="bright_black", err=True)
        
        for rank in display_ranks:
            if rank in base_taxon_reads:
                o_r = base_taxon_reads[rank]
                r_r = retained_taxon_reads.get(rank, 0.0)
                pct = (r_r / o_r * 100) if o_r > 0 else 0
                click.secho(f"{rank.capitalize():<16} {int(o_r):<18,} {int(r_r):<18,} {pct:.2f}%", err=True)
        click.secho("-" * 80, fg="bright_black", err=True)
        click.secho(" This table sums direct taxon assignments for each rank. Every classified read", fg="bright_black", italic=True, err=True)
        click.secho(" is counted exactly once.", fg="bright_black", italic=True, err=True)
        click.echo("", err=True)

    click.secho("[ 6 ] Final Table Shape", fg="cyan", bold=True, err=True)
    click.secho("-" * 80, fg="bright_black", err=True)
    
    row_str = f"{final_taxa_count}"
    parts = []
    if produced_synthetic:
         parts.append("1 synthetic 'Filtered' row")
    if has_unclass:
         parts.append("1 unclassified row")
    
    if parts:
        row_str += f" (incl. {', and '.join(parts)})"
        
    click.secho(f"Rows (Taxa)           : {row_str}", err=True)
    click.secho(f"Columns (Samples)     : {num_sample_cols}\n", err=True)
    click.secho("="*80 + "\n", fg="bright_black", err=True)


def _aggregate_reads_by_rank(df: pl.DataFrame, tree: JolTree, col_name: str = "clade_reads") -> Dict[str, float]:
    """Helper to sum specified read column for each canonical rank."""
    # 1. Sum across samples for each node. 
    # CRITICAL: Cast to UInt64 before sum to prevent 32-bit integer overflow.
    node_sums = df.group_by("t_id").agg(pl.col(col_name).cast(pl.UInt64).sum())
    
    # 2. Extract to numpy
    tids = node_sums["t_id"].to_numpy()
    reads = node_sums[col_name].to_numpy().astype(np.float64)
    
    counts = {}
    if len(tids) == 0:
        return counts

    # 3. Vectorized rank lookup
    # Map input TaxIDs to internal tree indices
    valid_idx = tree._get_indices(tids)
    found_mask = valid_idx != -1
    
    # Filter to only taxa found in the tree
    final_indices = valid_idx[found_mask]
    final_reads = reads[found_mask]
    
    if len(final_indices) > 0:
        # Retrieve ranks for all found nodes in one go
        ranks = tree.ranks[final_indices]
        
        # Accumulate sums by rank name
        for r_idx, read_count in zip(ranks, final_reads):
            r_name = tree.rank_names[r_idx]
            counts[r_name] = counts.get(r_name, 0.0) + read_count
            
    return counts


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
        keep_filtered  : If True (taxon/canonical modes only), retains filtered reads in a 
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
        ValueError        : If required parameters are missing for the selected mode, or if 
                            keep_filtered is used with an incompatible mode.
        FileNotFoundError : If the input file does not exist.
        FileExistsError   : If output_file exists and overwrite is False.
    """
    click.secho("Initiating table generation...", fg="cyan", err=True)

    if not overwrite and output_file.exists():
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace it.")

    # 0. Early Validations (Point 1 and 2)
    if not dry_run:
        check_write_permission(output_file)
        
    if cores:
        import os
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    # 1. Validate Parquet and Read Metadata via JSON Companion
    required_cols = ["sample_id", "t_id", "taxon_reads"]
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
    if keep_filtered:
        click.secho("Calculating baseline totals for keep-filtered logic...", fg="cyan", err=True)
            
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
    # The taxonomy tree is now required for ALL modes to support dynamic clade filtering,
    # recursive filtering via calc_clade_sum, and rank-based audit reports.
    if not taxonomy_dir:
        raise ValueError("Taxonomy directory is required for table generation.")
        
    click.secho("Loading JolTax taxonomy tree...", fg="cyan", err=True)
    tree = JolTree.load(str(taxonomy_dir))
        
    if clade_filter is not None:
        click.secho(f"Applying clade filter for TaxID {clade_filter}...", fg="cyan", err=True)
        clade_taxids = tree.get_clade(clade_filter)
        lf = lf.filter(pl.col("t_id").is_in(clade_taxids))
        
    if not keep_unclassified and mode != "canonical":
        click.secho("Filtering out unclassified reads...", fg="cyan", err=True)
        lf = lf.filter(pl.col("t_id") != 0)
        
    # 5. Dynamic Clade Math & Filtering
    click.secho("Applying dynamic recursive filtering and calculating clades...", fg="cyan", err=True)
    target_cols = ["t_id", "sample_id", "taxon_reads"]
    if data_standard == "SWEBITS":
        target_cols.extend(["year", "week"])
        
    input_df = lf.select(target_cols).collect()
    
    if data_standard == "SWEBITS":
        # Consolidate SWEBITS samples into periods
        input_df = input_df.with_columns(
            (pl.col("year").cast(pl.String) + "_" + pl.col("week").cast(pl.String).str.pad_start(2, "0")).alias("period")
        ).drop("sample_id").rename({"period": "sample_id"})
        
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
        # Re-apply clade filter because calc_clade_sum automatically adds all ancestors up to the root
        # to ensure mathematical integrity, but we don't want to output ancestors outside the requested clade.
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
        # Canonical logic needs both taxon and clade reads for the remaining active nodes
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
    baseline_taxa_count = baseline_df.select("t_id").unique().height
    base_tids = baseline_df.select("t_id").unique()["t_id"].to_list()
    
    sample_cols = [c for c in table.columns if c != "t_id"]
    
    # Baseline reads is the sum of taxon_reads
    baseline_reads = 0
    if not proportions and sample_cols:
        baseline_reads = baseline_df.select(pl.col("taxon_reads").sum()).item()
    
    # Retained reads
    retained_reads = 0
    if not proportions and sample_cols:
        retained_reads = filtered_df.select(pl.col("taxon_reads").sum()).item()
        
    # Calculate rank-based read retention for audit report
    # We do this by grouping the long-format baseline/filtered dataframes by rank
    # Note: calc_clade_sum ensures a node's clade_reads perfectly sum its surviving branch.
    # To get total reads in a rank, we just sum clade_reads for all nodes of that rank.
    # We'll pass the baseline_df and filtered_df to the audit report function.
    
    # 7. Composition Preservation
    produced_synthetic = False
    if keep_filtered and sample_cols:
        click.secho("Applying keep-filtered logic to preserve mass balance...", fg="cyan", err=True)
        filtered_row = {"t_id": FILTERED_TID}
        has_filtered = False
        
        # We need the original total reads to be safe against modes that drop reads intrinsically (like canonical NCA).
        # We calculate the sum of surviving taxon_reads per sample from the base filtered_df.
        # This is mode-independent and avoids the double-counting issue in 'clade' mode.
        surviving_totals = filtered_df.group_by("sample_id").agg(pl.col("taxon_reads").sum().alias("survived"))
        surviving_map = dict(zip(surviving_totals["sample_id"].to_list(), surviving_totals["survived"].to_list()))

        for c in sample_cols:
            survived = surviving_map.get(c, 0)
            original_total = true_totals.get(c, survived)
            diff = original_total - survived
            if diff < 0: diff = 0
            filtered_row[c] = diff
            if diff > 0:
                has_filtered = True
                
        if has_filtered:
            filtered_df_table = pl.DataFrame([filtered_row], schema=table.schema)
            table = pl.concat([table, filtered_df_table])
            produced_synthetic = True
            
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
            
    # 10. Generate Audit Report
    excl_count = len(excluded_ids) if exclude_samples else 0
    phantom_count = len(phantom_ids) if exclude_samples else 0
    actual_excluded = excl_count - phantom_count
    
    base_clade_dict = None
    retained_clade_dict = None
    base_taxon_dict = None
    retained_taxon_dict = None
    if tree and sample_cols and not proportions:
        base_clade_dict = _aggregate_reads_by_rank(baseline_df, tree, "clade_reads")
        retained_clade_dict = _aggregate_reads_by_rank(filtered_df, tree, "clade_reads")
        base_taxon_dict = _aggregate_reads_by_rank(baseline_df, tree, "taxon_reads")
        retained_taxon_dict = _aggregate_reads_by_rank(filtered_df, tree, "taxon_reads")
    
    if dry_run:
        _print_audit_report(
            dry_run=dry_run,
            input_name=input_parquet.name,
            total_samples=len(all_ids),
            actual_excluded=actual_excluded,
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
        click.secho("Dry-run complete. Exiting without saving.", fg="yellow", bold=True, err=True)
        return {
            "data_standard": data_standard,
            "active_samples": active_samples,
            "rows_output": table.height,
            "output_file": "DRY_RUN"
        }
        
    # 11. Output Generation
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

    # 12. Final Audit Report (Point 3)
    _print_audit_report(
        dry_run=dry_run,
        input_name=input_parquet.name,
        total_samples=len(all_ids),
        actual_excluded=actual_excluded,
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
        
    return {
        "data_standard": data_standard,
        "active_samples": active_samples,
        "rows_output": table.height,
        "output_file": str(output_file)
    }
