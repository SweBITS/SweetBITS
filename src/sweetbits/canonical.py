"""
sweetbits.canonical
Logic for calculating canonical remainders from clade counts.
"""

import polars as pl
import numpy as np
from typing import Optional
from joltax import JolTree
from joltax.constants import CANONICAL_RANKS

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
    
    # Track original sample_id dtype (it might be Categorical)
    orig_sample_dtype = df.schema["sample_id"]
    
    result = result_wide.unpivot(index="t_id", variable_name="sample_id", value_name="val")
    result = result.with_columns(
        pl.col("sample_id").cast(orig_sample_dtype),
        pl.col("val").cast(pl.UInt32)
    )
    
    if keep_unclassified and clade_filter is None and 0 not in rem_tids:
        unclass_lf = df.filter(pl.col("t_id") == 0).group_by(["sample_id", "t_id"]).agg(pl.col("clade_reads").sum().alias("val"))
        result = pl.concat([result, unclass_lf.select(["t_id", "sample_id", "val"])])

    if not keep_unclassified:
        result = result.filter(pl.col("t_id") != 0)
        
    return result
