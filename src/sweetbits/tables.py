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
from sweetbits.metadata import get_standard_metadata, write_parquet_with_metadata, read_parquet_metadata, validate_sweetbits_parquet

logger = logging.getLogger(__name__)

def calculate_canonical_remainders(
    df: pl.DataFrame,
    tree: JolTree,
    keep_unclassified: bool = False,
    clade_filter: Optional[int] = None
) -> pl.DataFrame:
    """
    Calculates taxonomic remainders by pushing counts up to the nearest canonical ancestor.

    This implements the NCA (Nearest Canonical Ancestor) aggregation algorithm.
    It solves the 'double-counting' problem by ensuring reads are attributed 
    only to the most specific standard rank available within the requested scope.

    Args:
        df                : Long-format DataFrame with 't_id', 'sample_id', and 'clade_reads'.
        tree              : The loaded JolTree taxonomy cache.
        keep_unclassified : Whether to explicitly include/calculate TaxID 0.
        clade_filter      : Optional TaxID used to define the root of the calculation.

    Returns:
        A long-format DataFrame with columns ['t_id', 'sample_id', 'val'].

    Raises:
        ValueError        : If clade_filter is provided but is not a canonical rank.
        RuntimeError      : If the mass balance check fails for any sample.
    """
    num_total_nodes = len(tree._index_to_id)
    
    # 1. SCOPE DEFINITION
    # Establish calculation boundaries. We strictly enforce canonical ranks for filters 
    # to ensure mathematical consistency. The entry/exit times allow us to create 
    # a fast boolean mask to isolate the relevant subtree.
    global_root_idx = 0 
    if clade_filter:
        rank = tree.get_rank(clade_filter)
        allowed_ranks = set(CANONICAL_RANKS) | {tree.top_rank}
        if rank not in allowed_ranks:
            raise ValueError(
                f"Clade filter TaxID {clade_filter} has rank '{rank}', which is not a canonical rank."
            )
        calc_root_idx = tree._get_indices(np.array([clade_filter]))[0]
        entry = tree.entry_times[calc_root_idx]
        exit = tree.exit_times[calc_root_idx]
        in_scope = (tree.entry_times >= entry) & (tree.entry_times < exit)
    else:
        calc_root_idx = global_root_idx
        in_scope = np.ones(num_total_nodes, dtype=bool)

    # 2. NCA TARGET MAPPING
    # Every node in the tree must know: "Who is my nearest canonical parent?". 
    # We use JolTax depths to ensure that we always pick the most specific rank 
    # (e.g., Species over Genus). Nodes without a canonical target default to the 
    # calculation root.
    target_map = np.full(num_total_nodes, -1, dtype=np.int32)
    depths = np.full(num_total_nodes, -1, dtype=np.int32)
    allowed_ranks = set(CANONICAL_RANKS) | {tree.top_rank}
    
    for rank, map_arr in tree.canonical_maps.items():
        if rank in allowed_ranks:
            valid_anc = (map_arr != -1) & in_scope[map_arr]
            anc_depths = np.full(num_total_nodes, -1, dtype=np.int32)
            anc_depths[valid_anc] = tree.depths[map_arr[valid_anc]]
            mask = valid_anc & (anc_depths > depths)
            depths[mask] = anc_depths[mask]
            target_map[mask] = map_arr[mask]
            
    target_map[(target_map == -1) & in_scope] = calc_root_idx
    
    # 3. DATA PREPARATION & ACTIVE NODE IDENTIFICATION
    # Pivot the input data into a matrix first. This naturally deduplicates TaxIDs 
    # and safely aligns all samples into fixed columns for vectorized math.
    matrix_df = df.pivot(values="clade_reads", index="t_id", on="sample_id", aggregate_function="sum").fill_null(0)
    sample_names = [c for c in matrix_df.columns if c != "t_id"]
    num_samples = len(sample_names)
    
    matrix_tids = matrix_df["t_id"].to_numpy()
    matrix_indices = tree._get_indices(matrix_tids)
    
    # Filter to valid indices strictly within our requested taxonomic scope
    valid_matrix_mask = (matrix_indices != -1) & (matrix_indices < num_total_nodes) & in_scope[matrix_indices]
    valid_tree_indices = matrix_indices[valid_matrix_mask]
    
    # Identify the specific canonical nodes that will appear in our final output.
    # We map every valid input node to its Nearest Canonical Ancestor (NCA).
    active_canonical_indices = np.unique(target_map[valid_tree_indices])
    
    # 4. AGGREGATION SETUP (THE 'VOTING' PATH)
    # Define the subtraction flow: each canonical node votes its entire clade 
    # value into its parent's NCA bucket. np.arange maps these tree indices back 
    # to local, 0-indexed positions in our results matrix.
    is_not_root = active_canonical_indices != calc_root_idx
    active_canonical_subset = active_canonical_indices[is_not_root]
    
    parent_indices = tree.parents[active_canonical_subset]
    contribution_targets = target_map[parent_indices]
    
    tree_to_active_pos = np.full(num_total_nodes, -1, dtype=np.int32)
    tree_to_active_pos[active_canonical_indices] = np.arange(len(active_canonical_indices))
    
    agg_targets = tree_to_active_pos[contribution_targets]
    agg_sources = np.where(is_not_root)[0]
    
    # Filter out aggregations where the target isn't in our active set
    # (In well-formed Kraken data, the target ancestor will always be present).
    valid_agg = agg_targets != -1
    agg_targets, agg_sources = agg_targets[valid_agg], agg_sources[valid_agg]
    
    # 5. VECTORIZED MATRIX POPULATION
    # We only pull clade_reads for nodes that ARE canonical; non-canonical reads 
    # are inferred dynamically during the subtraction phase from the parent's remainder.
    counts_matrix = np.zeros((len(active_canonical_indices), num_samples), dtype=np.int64)
    target_positions = tree_to_active_pos[valid_tree_indices]
    
    # Logical safeguard: Row must map to an active position and be an actual canonical node
    active_mask = (target_positions != -1) & (valid_tree_indices == active_canonical_indices[target_positions])
    
    final_target_positions = target_positions[active_mask]
    source_rows = np.where(valid_matrix_mask)[0][active_mask]
    
    # Bulk assignment: move all relevant clade_reads into counts_matrix in one step
    raw_counts = matrix_df.drop("t_id").to_numpy()
    counts_matrix[final_target_positions, :] = raw_counts[source_rows, :]
    
    # 6. VECTORIZED MATRIX SUBTRACTION
    # np.add.at performs buffered addition, ensuring all children are summed 
    # correctly into their parent's bucket even if multiple children share a parent.
    # Remainder = Total Clade - Sum of Canonical Child Clades.
    remainders = np.zeros((len(active_canonical_indices), num_samples), dtype=np.int64)
    for j in range(num_samples):
        sample_clade_counts = counts_matrix[:, j]
        child_sums = np.zeros(len(active_canonical_indices), dtype=np.int64)
        np.add.at(child_sums, agg_targets, sample_clade_counts[agg_sources])
        remainders[:, j] = sample_clade_counts - child_sums
        
    # 7. MASS BALANCE AUDIT
    # The absolute invariant: Sum of taxon_reads in scope must equal the Sum of 
    # standardized remainders. This verifies no reads were lost or gained.
    # We filter using only the subset of TaxIDs actually present in the input.
    in_scope_tids = matrix_tids[valid_matrix_mask]
    ground_truth = df.filter(pl.col("t_id").is_in(in_scope_tids)).group_by("sample_id").agg(pl.col("taxon_reads").sum().alias("total"))
    
    for j, sid in enumerate(sample_names):
        gt_row = ground_truth.filter(pl.col("sample_id") == sid)
        expected_total = gt_row["total"].sum() if not gt_row.is_empty() else 0
        actual_total = remainders[:, j].sum()
        if actual_total != expected_total:
            raise RuntimeError(
                f"Mass balance check failed for sample '{sid}'. "
                f"Ground truth (Sum of taxon_reads in scope) = {expected_total}, "
                f"Standardized total (Sum of remainders) = {actual_total}."
            )

    # 8. RECONSTRUCTION
    # Convert results back to long-format using Polars Unpivot (Melt) for 
    # maximum performance. Final join restores unclassified data if requested.
    rem_tids = tree._index_to_id[active_canonical_indices]
    result_wide = pl.from_numpy(remainders, schema=sample_names).with_columns(t_id = pl.Series(rem_tids).cast(pl.UInt32))
    result = result_wide.unpivot(index="t_id", variable_name="sample_id", value_name="val")
    
    if keep_unclassified and clade_filter is None and 0 not in rem_tids:
        unclass_lf = df.filter(pl.col("t_id") == 0).group_by(["sample_id", "t_id"]).agg(pl.col("clade_reads").sum().alias("val"))
        result = pl.concat([result, unclass_lf.select(["t_id", "sample_id", "val"])])

    if not keep_unclassified:
        result = result.filter(pl.col("t_id") != 0)
        
    return result

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

    This function automatically detects the 'data_standard' from the input file's
    metadata to decide how to format columns (YYYY_WW vs Sample ID). It supports
    filtering by clade, sample exclusion, and quality thresholds.

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

    Returns:
        A dictionary containing processing statistics:
        - 'data_standard'  : Detected standard (SWEBITS/GENERIC).
        - 'active_samples' : Number of samples included in the output.
        - 'rows_output'    : Number of taxa in the final table.
        - 'output_file'    : Path to the saved result.

    Raises:
        ValueError        : If required parameters are missing for the selected mode.
        FileNotFoundError : If the input file does not exist.
    """
    # 1. Validate Parquet and Read Metadata
    required_cols = ["sample_id", "t_id", "clade_reads", "taxon_reads"]
    metadata = validate_sweetbits_parquet(input_parquet, expected_type="REPORT_PARQUET", required_columns=required_cols)
    data_standard = metadata.get("data_standard", "GENERIC")
    
    lf = pl.scan_parquet(input_parquet)
    
    # 2. Filtering Samples
    # Execute ONE quick scan of just the sample IDs to power validation and math
    existing_samples_df = lf.select("sample_id").unique().collect()
    all_ids = set(existing_samples_df["sample_id"].to_list())
    
    if exclude_samples:
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

    # 3. Load Taxonomy Tree
    tree = None
    if mode == "canonical" or clade_filter is not None:
        if not taxonomy_dir:
            raise ValueError(f"Taxonomy directory is required for mode '{mode}' or clade filtering.")
        tree = JolTree.load(str(taxonomy_dir))
        
    if clade_filter is not None:
        clade_taxids = tree.get_clade(clade_filter)
        lf = lf.filter(pl.col("t_id").is_in(clade_taxids))
        
    if not keep_unclassified and mode != "canonical":
        lf = lf.filter(pl.col("t_id") != 0)
        
    # 4. Aggregation based on Mode
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

    # 5. Handle Column Naming based on Standard
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
    
    # 6. Final Quality Filters
    sample_cols = [c for c in table.columns if c != "t_id"]
    if sample_cols:
        if min_observed > 0:
            # Idiomatic Polars: evaluate directly inside the filter context
            table = table.filter(
                pl.sum_horizontal([(pl.col(c) > 0) for c in sample_cols]) >= min_observed
            )
            
        if min_reads > 0:
            table = table.filter(
                pl.max_horizontal(sample_cols) >= min_reads
            )
        
    # 7. Output Generation
    ext = output_file.suffix.lower()
    if ext == ".parquet":
        meta = get_standard_metadata("RAW_TABLE", source_path=input_parquet, sorting="t_id", data_standard=data_standard)
        write_parquet_with_metadata(table, output_file, meta)
    elif ext == ".csv":
        table.write_csv(output_file)
    elif ext == ".tsv":
        table.write_csv(output_file, separator="\t")
    else:
        raise ValueError(f"Unsupported output format: {ext}")
        
    return {
        "data_standard": data_standard,
        "active_samples": active_samples,
        "rows_output": table.height,
        "output_file": str(output_file)
    }
