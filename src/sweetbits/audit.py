"""
sweetbits.audit
Logic for generating and printing validation audit reports.
"""

import click
import polars as pl
import numpy as np
from typing import Optional, List, Dict
from joltax import JolTree
from joltax.constants import CANONICAL_RANKS

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

def print_audit_report(
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
