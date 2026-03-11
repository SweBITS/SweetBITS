"""
sweetbits.math
High-performance, vectorized math operations for taxonomy data.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Tuple
from joltax import JolTree

def calc_clade_sum(
    df: pl.DataFrame, 
    tree: JolTree, 
    min_reads: int = 0, 
    min_observed: int = 0,
    keep_filtered: bool = False
) -> Tuple[pl.DataFrame, np.ndarray]:
    """
    Computes hierarchical clade counts dynamically from direct taxon counts using 
    a highly optimized 'Level-Up Voting' algorithm.
    
    This function processes the taxonomic tree layer-by-layer from leaves to root.
    At each layer, nodes are evaluated against quality thresholds. Surviving nodes
    vote their clade sums up to their parents. Failed nodes are pruned recursively, 
    purifying the lineage signals above them.

    Args:
        df               : Long-format DataFrame with 't_id', 'sample_id', and 'taxon_reads'.
        tree             : The loaded JolTree taxonomy cache.
        min_reads        : Minimum maximum read count across all samples to survive.
        min_observed     : Minimum number of non-zero samples to survive.
        keep_filtered    : If True, returns a sum of pruned taxon reads for mass balance.

    Returns:
        A tuple containing:
        - A new DataFrame with updated 'taxon_reads' (orphans removed) and new 'clade_reads'.
        - A NumPy array of reads placed into the synthetic bin per sample (if keep_filtered).
    """
    # 1. Pivot the input data into a dense matrix (t_id x samples)
    matrix_df = df.pivot(
        values="taxon_reads", 
        index="t_id", 
        on="sample_id", 
        aggregate_function="sum"
    ).fill_null(0)
    
    sample_cols = [c for c in matrix_df.columns if c != "t_id"]
    if not sample_cols:
        return df.with_columns(pl.col("taxon_reads").alias("clade_reads")), np.array([])

    # Extract initial IDs
    initial_tids = matrix_df["t_id"].to_numpy()
    
    # 1.5 Expand Matrix to Include All Ancestors
    # In sparse datasets (only taxon_reads > 0 are present), internal nodes might be missing.
    # We must add them with 0 reads so they can receive "votes" from their children.
    initial_indices = tree._get_indices(initial_tids)
    valid_initial = initial_indices[initial_indices != -1]
    
    # Fast vectorized ancestor discovery
    num_total_nodes = len(tree._index_to_id)
    is_active = np.zeros(num_total_nodes, dtype=bool)
    is_active[valid_initial] = True
    
    current_layer = valid_initial
    while len(current_layer) > 0:
        parents = tree.parents[current_layer]
        valid_parents = parents[(parents != -1) & (parents != current_layer)]
        new_parents = valid_parents[~is_active[valid_parents]]
        if len(new_parents) == 0:
            break
        is_active[new_parents] = True
        current_layer = new_parents
        
    all_active_indices = np.where(is_active)[0]
    all_active_tids = np.array([int(tree._index_to_id[idx]) for idx in all_active_indices], dtype=np.uint32)
    
    # Find which tids are new and need to be appended with 0s
    existing_set = set(initial_tids)
    new_tids = [tid for tid in all_active_tids if tid not in existing_set]
    
    if new_tids:
        # Create a zero-filled dataframe for missing ancestors
        zero_data = {"t_id": new_tids}
        for col in sample_cols:
            zero_data[col] = [0] * len(new_tids)
        zero_df = pl.DataFrame(zero_data, schema=matrix_df.schema)
        matrix_df = pl.concat([matrix_df, zero_df])

    # Extract NumPy arrays for fast processing (use .copy() to ensure writability)
    tids = matrix_df["t_id"].to_numpy()
    taxon_reads = matrix_df.select(sample_cols).to_numpy().copy()
    clade_reads = taxon_reads.copy()
    
    # Map input TaxIDs to internal tree indices
    indices = tree._get_indices(tids)
    
    # Handle taxa not found in the tree (e.g., Unclassified, Filtered, outdated IDs)
    # They don't participate in the tree math, but we keep their values static.
    valid_mask = indices != -1
    valid_indices = indices[valid_mask]
    
    # To efficiently map tree operations back to our matrix rows, we need a reverse lookup
    num_total_nodes = len(tree._index_to_id)
    tree_to_matrix = np.full(num_total_nodes, -1, dtype=np.int32)
    tree_to_matrix[valid_indices] = np.where(valid_mask)[0]
    
    synthetic_bin = np.zeros(len(sample_cols), dtype=np.float64)
    
    # 2. The Level-Up Voting Algorithm
    max_depth = int(np.max(tree.depths)) if len(tree.depths) > 0 else 0
    
    for d in range(max_depth, -1, -1):
        # Find all valid matrix rows corresponding to nodes exactly at the current depth
        active_matrix_rows = np.where(valid_mask & (tree.depths[indices] == d))[0]
        if len(active_matrix_rows) == 0:
            continue
            
        # Get the current clade matrix for this layer
        layer_clade = clade_reads[active_matrix_rows, :]
        
        # Evaluate thresholds
        max_reads = np.max(layer_clade, axis=1)
        observed = np.sum(layer_clade > 0, axis=1)
        
        survivors = (max_reads >= min_reads) & (observed >= min_observed)
        
        # Identify successes and failures in matrix coordinates
        survivor_rows = active_matrix_rows[survivors]
        failed_rows = active_matrix_rows[~survivors]
        
        # The Purge: failed nodes
        if len(failed_rows) > 0:
            if keep_filtered:
                synthetic_bin += np.sum(taxon_reads[failed_rows, :], axis=0)
            clade_reads[failed_rows, :] = 0
            taxon_reads[failed_rows, :] = 0
            
        # The Vote: surviving nodes push their clade sum to their parents
        if len(survivor_rows) > 0:
            # Map matrix rows back to tree indices
            survivor_tree_indices = indices[survivor_rows]
            parents = tree.parents[survivor_tree_indices]
            
            # Find parents that are valid, exist in our target matrix, and are NOT self-loops (root)
            valid_parents_mask = (parents != -1) & (parents != survivor_tree_indices)
            target_parents = parents[valid_parents_mask]
            source_rows_for_vote = survivor_rows[valid_parents_mask]
            
            target_matrix_rows = tree_to_matrix[target_parents]
            
            # We only push votes to parents that are physically represented in our matrix.
            # (Kraken data should ideally contain all ancestors, but if not, the vote stops).
            valid_targets_mask = target_matrix_rows != -1
            final_targets = target_matrix_rows[valid_targets_mask]
            final_sources = source_rows_for_vote[valid_targets_mask]
            
            if len(final_targets) > 0:
                np.add.at(clade_reads, final_targets, clade_reads[final_sources])

    # 3. Reconstruct the Dataframe directly to long format
    # This avoids a massive memory spike caused by Polars unpivot() and join() on huge matrices.
    # By using np.where on a boolean mask, we extract only the non-zero (or unclassified) cells.
    from sweetbits.utils import UNCLASSIFIED_TID
    
    # We must keep cells where clade_reads > 0 OR it's the unclassified node 
    # to maintain mathematical integrity for the audit reports and mass balance.
    # First, find the row index for unclassified
    unclass_row_mask = tids == UNCLASSIFIED_TID
    
    # Create a boolean mask of the entire matrix (True = keep)
    keep_mask = (clade_reads > 0)
    
    # Force the unclassified row to be kept entirely (all samples)
    if np.any(unclass_row_mask):
        keep_mask[unclass_row_mask, :] = True
        
    # Extract just the non-zero values to save memory
    flat_taxon = taxon_reads[keep_mask]
    flat_clade = clade_reads[keep_mask]
    
    # Find the original matrix coordinates for those values
    row_idx, col_idx = np.where(keep_mask)
    
    # Map coordinates back to actual IDs
    tids_filtered = tids[row_idx]
    
    samples_arr = np.array(sample_cols)
    samples_filtered = samples_arr[col_idx]
    
    result_df = pl.DataFrame({
        "t_id": pl.Series(tids_filtered, dtype=matrix_df.schema["t_id"]),
        "sample_id": pl.Series(samples_filtered, dtype=pl.Categorical),
        "taxon_reads": pl.Series(flat_taxon, dtype=pl.UInt32),
        "clade_reads": pl.Series(flat_clade, dtype=pl.UInt32)
    })
    
    return result_df, synthetic_bin
