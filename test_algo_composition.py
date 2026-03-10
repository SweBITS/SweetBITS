import numpy as np

# Tree: Root(0) -> Genus(1) -> SpecA(2), SpecB(3)
depths = np.array([0, 1, 2, 2])
parents = np.array([-1, 0, 1, 1])

# Initial taxon reads for 1 sample: 
# SpecA = 10, SpecB = 50, Genus = 0, Root = 0
taxon_reads = np.array([[0], [0], [10], [50]])
clade_reads = taxon_reads.copy()

synthetic_bin = np.array([0])

max_depth = np.max(depths)
threshold = 20

for d in range(max_depth, -1, -1):
    mask = depths == d
    active_indices = np.where(mask)[0]
    if len(active_indices) == 0: continue
    
    max_reads = np.max(clade_reads[active_indices, :], axis=1)
    survivors = max_reads >= threshold
    
    survivor_indices = active_indices[survivors]
    failed_indices = active_indices[~survivors]
    
    # 1. Capture the failed reads for composition?
    # SpecA fails. It has 10 clade reads. 
    # Do we add 10 to the synthetic bin?
    for f_idx in failed_indices:
        synthetic_bin += clade_reads[f_idx]
    
    # Purge failed
    clade_reads[failed_indices, :] = 0
    taxon_reads[failed_indices, :] = 0
    
    # Vote up surviving
    valid_parents_mask = parents[survivor_indices] != -1
    valid_survivors = survivor_indices[valid_parents_mask]
    target_parents = parents[valid_survivors]
    np.add.at(clade_reads, target_parents, clade_reads[valid_survivors])

print("Final Taxon:\n", taxon_reads)
print("Synthetic Bin:", synthetic_bin)
