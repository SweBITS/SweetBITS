"""
sweetbits.canonical
Logic for calculating canonical remainders from clade counts and applying feature-based filtering.
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
    clade_filter: Optional[int] = None,
    min_reads: int = 0,
    min_observed: int = 0
) -> pl.DataFrame:
    """
    Calculates taxonomic remainders and performs bottom-up feature filtering (Remainder Bubbling).

    This implements the NCA (Nearest Canonical Ancestor) aggregation algorithm.
    It solves the 'double-counting' problem by ensuring reads are attributed 
    only to the most specific standard rank available within the requested scope.
    
    After raw remainders are calculated, they are evaluated against quality thresholds 
    (min_reads, min_observed) in a bottom-up sweep. Failed canonical taxa surrender 
    their reads to their NCA, ensuring 100% mass balance and guaranteeing all output 
    features are statistically significant.

    Args:
        df                : Long-format DataFrame with 't_id', 'sample_id', 'clade_reads', 'taxon_reads'.
                            (Should be unfiltered baseline data to ensure accurate raw remainders)
        tree              : The loaded JolTree taxonomy cache.
        keep_unclassified : Whether to explicitly include/calculate TaxID 0.
        clade_filter      : Optional TaxID used to define the root of the calculation.
        min_reads         : Minimum maximum read count across all samples for a canonical taxon to survive.
        min_observed      : Minimum number of samples a canonical taxon must appear in.

    Returns:
        A long-format DataFrame with columns ['t_id', 'sample_id', 'val'].

    Raises:
        ValueError        : If clade_filter is provided but is not a canonical rank.
        RuntimeError      : If the mass balance check fails for any sample.
    """
    num_total_nodes = len(tree._index_to_id)
    
    # 1. SCOPE DEFINITION
    # Dynamically find the true root index to avoid hardcoding bugs
    global_root_idx = tree._get_indices(np.array([1]))[0] 
    
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
    # Every node needs to know its Nearest Canonical Ancestor (NCA)
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
            
    # Nodes without a canonical target within the scope point to the root
    target_map[(target_map == -1) & in_scope] = calc_root_idx
    
    # 3. DATA PREPARATION & ACTIVE NODE IDENTIFICATION
    matrix_df = df.pivot(values="clade_reads", index="t_id", on="sample_id", aggregate_function="sum").fill_null(0)
    sample_names = [c for c in matrix_df.columns if c != "t_id"]
    num_samples = len(sample_names)
    
    if num_samples == 0:
        return df.select(["t_id", "sample_id", pl.col("clade_reads").alias("val")])
    
    matrix_tids = matrix_df["t_id"].to_numpy()
    matrix_indices = tree._get_indices(matrix_tids)
    
    valid_matrix_mask = (matrix_indices != -1) & (matrix_indices < num_total_nodes) & in_scope[matrix_indices]
    valid_tree_indices = matrix_indices[valid_matrix_mask]
    
    # Active canonical nodes are any nodes that serve as an NCA target for valid input taxa
    active_canonical_indices = np.unique(target_map[valid_tree_indices])
    
    # 4. AGGREGATION SETUP (THE 'VOTING' PATH)
    # Map each active canonical node to the row index it will occupy in our matrices
    tree_to_active_pos = np.full(num_total_nodes, -1, dtype=np.int32)
    tree_to_active_pos[active_canonical_indices] = np.arange(len(active_canonical_indices))

    # Determine where each active canonical node will subtract its mass from.
    # A canonical node subtracts its mass from the NCA of its parent.
    is_not_root = active_canonical_indices != calc_root_idx
    active_canonical_subset = active_canonical_indices[is_not_root]
    
    parent_indices = tree.parents[active_canonical_subset]
    
    # Safeguard against the root having no parent (-1)
    valid_parents_mask = parent_indices != -1
    active_canonical_subset = active_canonical_subset[valid_parents_mask]
    parent_indices = parent_indices[valid_parents_mask]
    
    contribution_targets = target_map[parent_indices]
    
    agg_targets = tree_to_active_pos[contribution_targets]
    agg_sources = tree_to_active_pos[active_canonical_subset]
    
    valid_agg = agg_targets != -1
    agg_targets, agg_sources = agg_targets[valid_agg], agg_sources[valid_agg]
    
    # 5. VECTORIZED MATRIX POPULATION
    # Extract only the clade_reads for the canonical nodes we care about
    counts_matrix = np.zeros((len(active_canonical_indices), num_samples), dtype=np.int64)
    target_positions = tree_to_active_pos[valid_tree_indices]
    
    active_mask = (target_positions != -1) & (valid_tree_indices == active_canonical_indices[target_positions])
    
    final_target_positions = target_positions[active_mask]
    source_rows = np.where(valid_matrix_mask)[0][active_mask]
    
    raw_counts = matrix_df.drop("t_id").to_numpy()
    counts_matrix[final_target_positions, :] = raw_counts[source_rows, :]
    
    # 6. VECTORIZED MATRIX SUBTRACTION (RAW REMAINDERS)
    # Remainder = Parent Clade - Sum(Canonical Child Clades)
    remainders = np.zeros((len(active_canonical_indices), num_samples), dtype=np.int64)
    for j in range(num_samples):
        sample_clade_counts = counts_matrix[:, j]
        child_sums = np.zeros(len(active_canonical_indices), dtype=np.int64)
        np.add.at(child_sums, agg_targets, sample_clade_counts[agg_sources])
        remainders[:, j] = sample_clade_counts - child_sums
        
    # 7. REMAINDER BUBBLING (FEATURE FILTERING)
    # Map each canonical node to its canonical parent for bubbling failed mass
    canonical_parent = np.full(len(active_canonical_indices), -1, dtype=np.int32)
    tree_parents_all = tree.parents[active_canonical_indices]
    valid_p_all = tree_parents_all != -1
    nca_of_parents = target_map[tree_parents_all[valid_p_all]]
    canonical_parent[valid_p_all] = tree_to_active_pos[nca_of_parents]
    
    c_depths = tree.depths[active_canonical_indices]
    max_depth = int(np.max(c_depths)) if len(c_depths) > 0 else 0
    calc_root_active_idx = tree_to_active_pos[calc_root_idx]
    
    # Track which nodes successfully pass the thresholds
    is_survivor = np.ones(len(active_canonical_indices), dtype=bool)
    
    # Sweep from the deepest leaves up to the root
    for d in range(max_depth, -1, -1):
        layer_indices = np.where(c_depths == d)[0]
        if len(layer_indices) == 0:
            continue
            
        layer_rem = remainders[layer_indices, :]
        max_reads = np.max(layer_rem, axis=1)
        observed = np.sum(layer_rem > 0, axis=1)
        
        # Test the node's current remainder mass against the thresholds
        survivors = (max_reads >= min_reads) & (observed >= min_observed)
        
        # The ultimate root of our calculation must survive to catch the final mass
        is_root_mask = (layer_indices == calc_root_active_idx)
        survivors = survivors | is_root_mask
        
        failed_local_indices = layer_indices[~survivors]
        
        # If nodes fail, they surrender their mass to their canonical parent
        if len(failed_local_indices) > 0:
            is_survivor[failed_local_indices] = False
            targets = canonical_parent[failed_local_indices]
            
            valid_push_mask = targets != -1
            push_sources = failed_local_indices[valid_push_mask]
            push_targets = targets[valid_push_mask]
            
            if len(push_sources) > 0:
                # Add the failed node's entire row to its parent's row
                np.add.at(remainders, push_targets, remainders[push_sources])
                
            # Erase the failed node
            remainders[failed_local_indices, :] = 0

    # 8. MASS BALANCE AUDIT
    # The sum of all remainders must perfectly equal the original taxon_reads in the tree
    in_scope_tids = matrix_tids[valid_matrix_mask]
    ground_truth = df.filter(pl.col("t_id").is_in(in_scope_tids)).group_by("sample_id").agg(pl.col("taxon_reads").sum().alias("total"))
    
    gt_dict = dict(zip(ground_truth["sample_id"].to_list(), ground_truth["total"].to_list()))
    
    for j, sid in enumerate(sample_names):
        expected_total = gt_dict.get(sid, 0)
        actual_total = remainders[:, j].sum()
        if actual_total != expected_total:
            raise RuntimeError(
                f"Mass balance check failed for sample '{sid}'. "
                f"Ground truth (Sum of taxon_reads in scope) = {expected_total}, "
                f"Standardized total (Sum of remainders) = {actual_total}."
            )

    # 9. RECONSTRUCTION
    # Keep only survivors that actually have mass (safety net against structural ghost nodes)
    survivor_indices = np.where(is_survivor & (np.sum(remainders, axis=1) > 0))[0]
    
    if len(survivor_indices) == 0:
        schema = {"t_id": pl.UInt32, "sample_id": df.schema["sample_id"], "val": pl.UInt32}
        return pl.DataFrame(schema=schema)
        
    final_remainders = remainders[survivor_indices, :]
    rem_tids = tree._index_to_id[active_canonical_indices[survivor_indices]]
    
    result_wide = pl.from_numpy(final_remainders, schema=sample_names).with_columns(t_id = pl.Series(rem_tids).cast(pl.UInt32))
    
    orig_sample_dtype = df.schema["sample_id"]
    
    # Fast unpivot
    result = result_wide.unpivot(index="t_id", variable_name="sample_id", value_name="val")
    result = result.with_columns(
        pl.col("sample_id").cast(orig_sample_dtype),
        pl.col("val").cast(pl.UInt32)
    )
    
    # Remove any stray zeroes generated during unpivoting sparse rows
    result = result.filter(pl.col("val") > 0)
    
    # Restore unclassified reads if required
    if keep_unclassified and clade_filter is None and 0 not in rem_tids:
        unclass_lf = df.filter(pl.col("t_id") == 0).group_by(["sample_id", "t_id"]).agg(pl.col("clade_reads").sum().alias("val"))
        result = pl.concat([result, unclass_lf.select(["t_id", "sample_id", "val"])])

    if not keep_unclassified:
        result = result.filter(pl.col("t_id") != 0)
        
    return result
