# SweetBITS Architectural Refactor: Dynamic Clade Math & Recursive Filtering

## 1. Core Objective
Migrate SweetBITS from a "static clade" architecture to a "dynamic clade" architecture. 
The fundamental unit of truth will become the `taxon_reads` (direct assignments). All hierarchical metrics (`clade_reads`, canonical remainders) will be calculated dynamically on-the-fly using the JolTax tree.

## 2. Architectural Changes
*   **Storage Reduction:** Remove the `clade_reads` column entirely from the `<REPORT_PARQUET>` specification and generation logic (`gather_reports`). This will significantly reduce Parquet file sizes and memory footprint during pivoting.
*   **Taxonomy Dependency:** Any command or mode that relies on hierarchical logic (e.g., `produce table -m clade`, `canonical` mode) will now strictly require the `-t / --taxonomy` flag.

## 3. The "Level-Up Voting" Algorithm
To efficiently calculate clade counts and apply thresholds, we will introduce a new core function: `calc_clade_sum`. 

This algorithm solves the problem of "sparse noisy children diluting parent saturation metrics" by applying filters recursively from leaf to root using a vectorized, single-pass strategy based on node depth.

### Algorithm Flow:
1.  **Initialization:** 
    *   Load `taxon_reads` matrix (rows = internal tree indices, columns = samples).
    *   Initialize `clade_reads` matrix as a direct copy of `taxon_reads`.
    *   Initialize a 1D `synthetic_bin` array for `--keep-composition` (length = num samples).
2.  **The Loop (Bottom-Up):**
    *   Iterate backwards from `max_depth` (the deepest leaves) down to `0` (the Root).
    *   At `current_depth`, mask all active nodes.
3.  **The Trial (Threshold Evaluation):**
    *   Evaluate the active nodes against `--min-reads` and `--min-observed` using their current `clade_reads`.
4.  **The Purge (Failures):**
    *   For nodes that **fail** the thresholds:
        *   If `--keep-composition` is true, add their entire `clade_reads` row to the `synthetic_bin`.
        *   Zero out their `clade_reads` row.
        *   Zero out their `taxon_reads` row (Lineage Eraser).
5.  **The Vote (Survivors):**
    *   For nodes that **pass** the thresholds:
        *   Add their `clade_reads` row to their parent's `clade_reads` row using `np.add.at`.
6.  **Cleanup:**
    *   After hitting `depth = 0`, discard any row where `clade_reads == 0`.

### Mathematical Guarantees
*   **Lineage Integrity:** Because absolute metrics strictly increase as they move up the tree, a child passing the threshold guarantees the parent will pass. No "dead internal nodes with living children" can exist.
*   **Mass Balance:** Failed nodes dump exactly their terminal `clade_reads` into the synthetic bin *before* being zeroed. This perfectly captures all discarded reads in the branch without double-counting.

## 4. The Read Retention Audit Report
With `clade_reads` calculated dynamically, the Audit Report (`--dry-run`) can be vastly improved.

We will calculate two sets of clade counts:
1.  **Baseline Clades:** Run `calc_clade_sum` with thresholds set to `0`.
2.  **Retained Clades:** Run `calc_clade_sum` with the user's requested thresholds.

By summing the `clade_reads` for all nodes belonging to a specific Canonical Rank (e.g., sum of all Genera), we can generate a powerful new table:

**[ 4 ] Read Retention by Rank**
```text
Rank             Original Reads     Retained Reads     Retention % 
--------------------------------------------------------------------------------
Domain           950,000            880,000            92.6%
Phylum           900,000            800,000            88.8%
...
Species          200,000            20,000             10.0%
```

## 5. Implementation Steps
- [ ] 0. **Version Bump:** Update `__version__` to `0.1.1` and bump `MINIMUM_COMPATIBLE_VERSION` to `0.1.1` in `metadata.py` to reject old `clade_reads` Parquet files.
- [ ] 1. **Update Ingestion:** Update `tests/` and `gather_reports` logic to stop generating/expecting `clade_reads`.
- [ ] 2. **Build `calc_clade_sum`:** Build and test the vectorized `calc_clade_sum` function in a new utility module (e.g., `sweetbits.math`). Ensure the "vote" (adding `clade_reads` to the parent) happens in *every* step of the layer-by-layer loop for surviving nodes.
- [ ] 3. **Refactor Table Generation:** Refactor `tables.py` to use `calc_clade_sum` for `clade` and `canonical` mode generation.
- [ ] 4. **Audit Report:** Implement the "Read Retention by Rank" logic into the `_print_audit_report` function.
- [ ] 5. **Documentation:** Update CLI documentation and validation checks (enforce `-t` for clade modes).