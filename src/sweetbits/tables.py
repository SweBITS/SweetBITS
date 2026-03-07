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
from sweetbits.utils import parse_sample_id, load_sample_id_list
from sweetbits.metadata import get_standard_metadata, write_parquet_with_metadata

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
    keep_unclassified: bool = False
) -> Dict[str, Any]:
    """
    Generates a wide-format abundance table from a merged REPORT_PARQUET file.

    This function handles three primary abundance modes:
    - 'taxon': Raw reads assigned directly to a TaxID.
    - 'clade': Cumulative reads for a taxon and all descendants (redundant).
    - 'canonical': Canonical remainders. Essentially taxon mode but where reads 
                   between canonical ranks have been pushed up to the nearest 
                   canonical ancestor (NCA). Ensures mass balance.

    Args:
        input_parquet: Path to the merged REPORT_PARQUET file.
        output_file: Path where the table will be saved (.csv, .tsv, or .parquet).
        mode: Abundance calculation mode ('taxon', 'clade', 'canonical').
        taxonomy_dir: Path to JolTax cache directory (required for 'canonical' or clade filter).
        exclude_samples: Optional path to a text file containing Sample IDs to exclude.
        min_observed: Minimum number of samples a taxon must appear in.
        min_reads: Minimum maximum read count across all samples for a taxon.
        clade_filter: Optional TaxID to restrict output to a specific clade.
        keep_unclassified: Whether to include TaxID 0 in the output.

    Returns:
        A summary dictionary containing 'active_samples', 'rows_output', 
        'output_file', and 'status'.

    Raises:
        ValueError: If required parameters (like taxonomy_dir) are missing for the selected mode.
    """
    # 1. Initialize LazyFrame
    lf = pl.scan_parquet(input_parquet)
    
    # 2. Filtering Samples
    if exclude_samples:
        excluded_ids = load_sample_id_list(exclude_samples)
        
        # Check if excluded IDs actually exist in the dataset
        all_ids = set(lf.select("sample_id").unique().collect()["sample_id"].to_list())
        phantom_ids = [eid for eid in excluded_ids if eid not in all_ids]
        
        if phantom_ids:
            click.secho(
                f"Warning: {len(phantom_ids)} sample IDs in exclusion file were not found in the dataset "
                f"(e.g., {phantom_ids[0]}). Please check for typos.",
                fg="yellow", err=True
            )
            
        lf = lf.filter(~pl.col("sample_id").is_in(excluded_ids))
        
    # Get active sample count without loading full data
    active_samples = lf.select("sample_id").unique().collect().height
    
    # Issue warning if filtering threshold is very high relative to sample count
    if min_observed > (active_samples / 2) and active_samples > 0:
        click.secho(
            f"Warning: --min-observed ({min_observed}) is more than 50% of active samples ({active_samples}). "
            "This may result in an empty or very sparse table.", 
            fg="yellow", err=True
        )

    # 3. Load Taxonomy Tree if needed
    tree = None
    if mode in ["clade", "canonical"] or clade_filter is not None:
        if not taxonomy_dir:
            raise ValueError(f"Taxonomy directory is required for mode '{mode}' or clade filtering.")
        tree = JolTree.load(str(taxonomy_dir))
        
    # 3a. Apply Taxonomic Clade Filter (Lazy)
    if clade_filter is not None:
        clade_taxids = tree.get_clade(clade_filter)
        lf = lf.filter(pl.col("t_id").is_in(clade_taxids))
        
    # 3b. Initial Unclassified Handling (re-evaluated in canonical mode)
    if not keep_unclassified and mode != "canonical":
        lf = lf.filter(pl.col("t_id") != 0)
        
    # 4. Aggregation based on Mode
    if mode == "taxon":
        pivot_df = lf.select(["t_id", "sample_id", "year", "week", "taxon_reads"]).collect()
        pivot_col = "taxon_reads"
    elif mode == "clade":
        pivot_df = lf.select(["t_id", "sample_id", "year", "week", "clade_reads"]).collect()
        pivot_col = "clade_reads"
    elif mode == "canonical":
        # --- CANONICAL REMAINDER LOGIC ---
        df = lf.select(["t_id", "sample_id", "year", "week", "clade_reads"]).collect()
        
        num_nodes = len(tree._index_to_id)
        
        # 4.1 Build NCA Target Map
        target_map = np.full(num_nodes, -1, dtype=np.int32)
        depths = np.full(num_nodes, -1, dtype=np.int32)
        relevant_ranks = set(CANONICAL_RANKS)
        if tree.top_rank == "domain":
            relevant_ranks.add("domain")
            relevant_ranks.discard("superkingdom")
        else:
            relevant_ranks.add("superkingdom")
            relevant_ranks.discard("domain")

        for rank, map_arr in tree.canonical_maps.items():
            if rank in relevant_ranks:
                valid_anc = (map_arr != -1)
                anc_depths = np.full(num_nodes, -1, dtype=np.int32)
                anc_depths[valid_anc] = tree.depths[map_arr[valid_anc]]
                mask = valid_anc & (anc_depths > depths)
                depths[mask] = anc_depths[mask]
                target_map[mask] = map_arr[mask]
        
        target_map[target_map == -1] = 0
        
        # 4.2 Identify Active Canonical Nodes
        input_tids = df["t_id"].unique().to_numpy()
        input_indices = tree._get_indices(input_tids)
        valid_input_mask = input_indices != -1
        
        ncas_of_input = target_map[input_indices[valid_input_mask]]
        active_canonical_indices = np.unique(ncas_of_input)
        
        # 4.3 Prepare for aggregation
        root_idx = 0 
        not_root_mask = active_canonical_indices != root_idx
        
        parent_indices = tree.parents[active_canonical_indices[not_root_mask]]
        contribution_targets = target_map[parent_indices]
        
        tree_to_active = np.full(num_nodes, -1, dtype=np.int32)
        tree_to_active[active_canonical_indices] = np.arange(len(active_canonical_indices))
        
        target_active_indices = tree_to_active[contribution_targets]
        source_active_indices = np.where(not_root_mask)[0]
        
        valid_targets_mask = target_active_indices != -1
        agg_targets = target_active_indices[valid_targets_mask]
        agg_sources = source_active_indices[valid_targets_mask]
        
        # 4.4 Calculate remainders sample-by-sample
        matrix_df = df.pivot(
            values="clade_reads",
            index="t_id",
            on="sample_id",
            aggregate_function="sum"
        ).fill_null(0)
        
        sample_names = [c for c in matrix_df.columns if c != "t_id"]
        counts_matrix = matrix_df[sample_names].to_numpy()
        matrix_tids = matrix_df["t_id"].to_numpy()
        matrix_indices = tree._get_indices(matrix_tids)
        
        idx_to_matrix_pos = np.full(num_nodes, -1, dtype=np.int32)
        idx_to_matrix_pos[matrix_indices[matrix_indices != -1]] = np.arange(np.sum(matrix_indices != -1))
        
        active_in_input_pos = idx_to_matrix_pos[active_canonical_indices]
        found_mask = active_in_input_pos != -1
        
        remainders = np.zeros((len(active_canonical_indices), len(sample_names)), dtype=np.int64)
        
        for j in range(len(sample_names)):
            sample_clade_counts = np.zeros(len(active_canonical_indices), dtype=np.int64)
            sample_clade_counts[found_mask] = counts_matrix[active_in_input_pos[found_mask], j]
            
            child_sums = np.zeros(len(active_canonical_indices), dtype=np.int64)
            np.add.at(child_sums, agg_targets, sample_clade_counts[agg_sources])
            
            remainders[:, j] = sample_clade_counts - child_sums
            
        # 4.5 Convert back to long-format for pivoting
        rem_tids = tree._index_to_id[active_canonical_indices]
        tid_col, sample_col, val_col = [], [], []
        
        for i, tid in enumerate(rem_tids):
            for j, sid in enumerate(sample_names):
                tid_col.append(tid)
                sample_col.append(sid)
                val_col.append(remainders[i, j])
        
        if keep_unclassified and clade_filter is None:
            if 0 not in rem_tids:
                for sid in sample_names:
                    unclass_val = 0
                    row = df.filter((pl.col("t_id") == 0) & (pl.col("sample_id") == sid))
                    if not row.is_empty(): unclass_val = row["clade_reads"].sum()
                    tid_col.append(0); sample_col.append(sid); val_col.append(unclass_val)

        pivot_df = pl.DataFrame({"t_id": tid_col, "sample_id": sample_col, "val": val_col})
        pivot_df = pivot_df.with_columns(pl.col("t_id").cast(pl.UInt32))
        
        if not keep_unclassified:
            pivot_df = pivot_df.filter(pl.col("t_id") != 0)
        
        # Re-attach sample metadata
        sample_meta = df.select(["sample_id", "year", "week"]).unique()
        pivot_df = pivot_df.join(sample_meta, on="sample_id")
        pivot_col = "val"

    # 5. Pivot to wide format (Columns: YYYY_WW)
    pivot_df = pivot_df.with_columns(
        (pl.col("year").cast(pl.String) + "_" + pl.col("week").cast(pl.String).str.pad_start(2, "0")).alias("period")
    )
    
    table = pivot_df.pivot(
        values=pivot_col,
        index="t_id",
        on="period",
        aggregate_function="sum"
    ).fill_null(0).sort("t_id")
    
    # 6. Apply Occupancy and Depth Filters
    sample_cols = [c for c in table.columns if c != "t_id"]
    if not sample_cols:
        table = pl.DataFrame(schema={"t_id": pl.UInt32})
    else:
        if min_observed > 0:
            obs_count = table.select([
                pl.sum_horizontal([pl.col(c) > 0 for c in sample_cols]).alias("count")
            ])["count"]
            table = table.filter(obs_count >= min_observed)
        if min_reads > 0:
            max_reads = table.select([
                pl.max_horizontal(sample_cols).alias("max_val")
            ])["max_val"]
            table = table.filter(max_reads >= min_reads)
        
    # 7. Output Generation
    ext = output_file.suffix.lower()
    if ext == ".parquet":
        metadata = get_standard_metadata("RAW_TABLE", source_path=input_parquet, sorting="t_id")
        write_parquet_with_metadata(table, output_file, metadata)
    elif ext == ".csv":
        table.write_csv(output_file)
    elif ext == ".tsv":
        table.write_csv(output_file, separator="\t")
    else:
        raise ValueError(f"Unsupported output format: {ext}")
        
    return {
        "active_samples": active_samples,
        "rows_output": table.height,
        "output_file": str(output_file)
    }
