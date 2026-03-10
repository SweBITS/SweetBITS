import numpy as np

# Simple tree mock
# Root (0) -> Parent (1) -> Leaf1 (2), Leaf2 (3)
# Depth: Root=0, Parent=1, Leaves=2

depths = np.array([0, 1, 2, 2])
parents = np.array([-1, 0, 1, 1])

# Initial taxon reads: Root=0, Parent=5, Leaf1=10, Leaf2=50
taxon_reads = np.array([[0], [5], [10], [50]])
clade_reads = taxon_reads.copy()

print("Initial Clade:\n", clade_reads)

max_depth = np.max(depths)

for d in range(max_depth, -1, -1):
    mask = depths == d
    active_indices = np.where(mask)[0]
    if len(active_indices) == 0: continue
    
    # Filter logic (e.g. min_reads >= 15)
    # Applied to CLADE reads of the current layer
    max_reads = np.max(clade_reads[active_indices, :], axis=1)
    survivors = max_reads >= 15
    
    survivor_indices = active_indices[survivors]
    failed_indices = active_indices[~survivors]
    
    # Purge failed
    clade_reads[failed_indices, :] = 0
    taxon_reads[failed_indices, :] = 0
    
    # Vote up surviving
    valid_parents_mask = parents[survivor_indices] != -1
    valid_survivors = survivor_indices[valid_parents_mask]
    target_parents = parents[valid_survivors]
    
    # Add clade values to parents
    np.add.at(clade_reads, target_parents, clade_reads[valid_survivors])

print("\nFinal Taxon:\n", taxon_reads)
print("\nFinal Clade:\n", clade_reads)
