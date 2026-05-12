# SweetBITS Feature Generation (`produce feature`)

The `feature` subcommand group provides specialized statistical engines for extracting taxonomic classification quality features. These features are designed to validate taxonomic presence and differentiate between true biological signals and false positives (e.g., contamination, database bias).

The generated features are typically output as a table indexed by `t_id`, which can then be joined to any SweetBITS abundance table using the `sweetbits annotate --metadata` command.

---

## 1. Unique Minimizer Correlations (`uniq-minimizer-corr`)
This engine validates taxonomic assignments by comparing observed unique minimizer coverage against a probabilistic expectation model based on sequencing depth.

### Overview
- **Logic:** Testing if the accumulation of evidence (unique minimizers) follows a "natural" distribution.
- **Requirement:** 8-column Kraken reports (`HYPERLOGLOG` format).
- **Scope:** Calculates features across multiple samples to determine stability.

### Feature Description
- **`mm_pearson_corr`**: Pearson correlation between observed and expected unique minimizer coverage across all samples. High values (e.g., > 0.8) indicate reliable evidence accumulation.
- **`mm_pearson_p`**: Two-tailed p-value for the Pearson correlation.
- **`mm_pearson_n`**: Number of samples used in the calculation ($n \ge 6$ required).
- **`mm_pearson_filtered_corr`**: Pearson correlation after removing top outliers (e.g., extremely high coverage samples) to improve robustness.
- **`mm_obs_cov_mean / _median`**: Distributional stats of the observed unique minimizer coverage.
- **`mm_obs_cov_cv`**: Coefficient of Variation (Stability) of the observed coverage.
- **`mm_obs_cov_p05 / _p95`**: 5th and 95th percentiles of observed coverage.

---

## 2. Global K-mer Evidence Features (`kmer-global`)
This engine pools all k-mer classification data across any number of samples to create a "Global Total" evidence profile for every species in the dataset.

#### A. Core Count Metrics (`kmers_global_..._count`)
- **`kmers_global_clade_count`**: Total k-mers classified to the target species or its descendants.
- **`kmers_global_classified_count`**: Total k-mers that received *any* taxonomic assignment.
- **`kmers_global_total_count`**: Absolute total of all k-mers (including unclassified).
- **`kmers_global_unclassified_count`**: Total number of k-mers that remained unclassified.
- **`kmers_global_lineage_count`**: Total classified k-mers hitting the taxonomic lineage (e.g., hitting the Genus but not the Species).
- **`kmers_global_root_count`**: Total k-mers hitting the Root (TaxID 1).
- **`kmers_global_misclassified_count`**: K-mers classified outside the species AND outside its lineage (potential noise).
- **`kmers_global_exclade_count`**: Total k-mers hitting anything *outside* the target species clade.

#### B. Evidence Ratios (`kmers_global_..._ratio`)
These features provide proportions normalized by either **Classified** k-mers or **Total** k-mers using the `XVSY_ratio` format. All ratios are strictly bounded `[0, 1]` with null-safe handling for division-by-zero.

**Relative to Classified K-mers:**
- **`kmers_global_cladeVSclassified_ratio`**: Proportion of classified evidence that correctly hits the clade.
- **`kmers_global_lineageVSclassified_ratio`**: Proportion of classified evidence hitting the lineage (high-level assignment).
- **`kmers_global_misclassifiedVSclassified_ratio`**: Proportion of classified evidence hitting unrelated taxa.
- **`kmers_global_rootVSclassified_ratio`**: Proportion of classified evidence at the root.

**Relative to Total K-mers (Confidence Scores):**
- **`kmers_global_cladeVStotal_ratio`**: The global Kraken 2 Confidence Score for this species.
- **`kmers_global_classifiedVStotal_ratio`**: Global classification rate for this taxon's reads.
- **`kmers_global_lineageVStotal_ratio`**: Proportion of all k-mers hitting the lineage.
- **`kmers_global_rootVStotal_ratio`**: Proportion of all k-mers hitting the root.
- **`kmers_global_misclassifiedVStotal_ratio`**: Global misclassification proportion.
- **`kmers_global_supportingVStotal_ratio`**: Proportion of all k-mers hitting Clade + Lineage.

**Internal Exclade Ratios:**
- **`kmers_global_rootVSexclade_ratio`**: Proportion of off-target hits that were pushed to the root.
- **`kmers_global_lineageVSexclade_ratio`**: Proportion of off-target hits that hit the lineage.
- **`kmers_global_excladeVStotal_ratio`**: Proportion of all k-mers hitting anything outside the clade.

#### C. Taxonomic Distance & Depth (Weighted Stats)
*Metrics: `mean`, `median`, `cv`, `p05`, `p95` for:*
- **`kmers_global_misclassified_dist_[STAT]`**: The number of nodes in the tree between the assigned species and the unrelated k-mer hit. Large distances suggest egregious misclassifications.
- **`kmers_global_misclassified_depth_[STAT]`**: The absolute depth in the tree of the taxa where misclassified k-mers hit.
- **`kmers_global_misclassified_relative_lca_depth_[STAT]`**: Depth of the Lowest Common Ancestor (LCA) relative to the target species depth. Values near 1.0 mean the "noise" is taxonomically close to the target.
- **`kmers_global_lineage_relative_depth_[STAT]`**: Where in the lineage k-mer hits are clustering (e.g., just above species vs. near root).

#### D. Top Hit Profiles
- **`kmers_global_misclassified_top5_names`**: Names of the top 5 unrelated taxonomic competitors.
- **`kmers_global_misclassified_top5_taxids`**: TaxIDs of the top 5 competitors.
- **`kmers_global_misclassified_top5_shares`**: Percentage of total misclassified k-mers held by each of the top 5 taxa.

---

## 3. Per-Sample K-mer Evidence Features (`kmer-sample`)
This engine generates a detailed temporal Parquet dataset containing classification features at a sample-by-sample resolution. This long-format table is essential for exploratory data analysis (EDA).

#### Core Metrics
- Calculates the same 8 absolute counts and 13 strictly bounded ratios as the global engine, but with the `kmers_sample_` prefix.
- Calculates sample-specific taxonomic distance, depth, and relative LCA depth distributions (`mean`, `median`, `cv`, `p05`, `p95`).
- Identifies the Top 3 competitor taxa (names, IDs, shares) per sample.

**Scalability (Divide and Conquer):** For large datasets, this engine supports a chunked workflow. Individual samples can be processed into intermediate Parquet "chunks" and subsequently unified using the `sweetbits collect feature-chunks` command. This ensures constant memory usage regardless of project size.

*Note on Machine Learning:* The raw `kmers_sample` Parquet file should **not** be passed directly into a standard GBM (like XGBoost) if your model expects a single row per species. Furthermore, string/list features (Top 3 competitors) must be dropped before training to prevent crashes or extreme memory bloat.

---

## 4. Inter-Sample Stability Features (`kmer-stability`)
This engine consumes the long-format output of `kmer-sample` to calculate variance metrics for each taxon. It provides the GBM with a critical signal to differentiate between stable biological presence and erratic false positives.

#### A. Occupancy & Presence
- **`kmers_stability_occupancy_ratio`**: The proportion of all project samples where this taxon appeared (Normalized [0, 1]).
- **`kmers_stability_[RATIO]_presence`**: The proportion of samples where a specific ratio was mathematically calculable (e.g., proportion of samples where `classified_count > 0`).

#### B. Stability Variance (`kmers_stability_[RATIO]_[STAT]`)
For each of the 13 core classification quality ratios, this engine calculates 5 distributional stats across all samples where the ratio was valid:
- `_mean`, `_median`, `_p05`, `_p95`.
- **`_cv`**: The Coefficient of Variation. A high CV in a critical ratio (like `cladeVStotal_ratio`) is a massive red flag that the taxon's evidence profile is unstable, often indicating stochastic contamination or systemic resolution collisions.

*Note:* To avoid mathematically invalid "means of means," stability metrics are **strictly restricted to proportions**. We do not calculate stability variance for raw counts or taxonomic distances.

---

### 5. Read Length Distribution Features (`reads_{global,sample}_...`)
These features quantify the physical fragmentation of the DNA assigned to a taxon. Ancient or highly degraded DNA typically shows shorter mean lengths and specific distribution shapes.

- **`reads_[scope]_total_count`**: Total reads analyzed.
- **`reads_[scope]_readlen_mean`**: Weighted mean read length.
- **`reads_[scope]_readlen_median`**: 50th percentile.
- **`reads_[scope]_readlen_p05`**: 5th percentile.
- **`reads_[scope]_readlen_p95`**: 95th percentile.
- **`reads_[scope]_readlen_cv`**: Coefficient of Variation.

---

### **Computational Complexity & Performance**
The generation of global k-mer features is a two-step process: **Ingestion** (parsing raw Kraken files) and **Extraction** (pooling data).

*   **Scaling:** Ingestion is the primary bottleneck. Based on the **Ljungbyhed dataset**, memory scales at **~4.8 GB per 1 GB** of compressed input, while runtime scales at **~3.5 minutes per 1 GB** (on 4 cores).
*   **Beyond File Size:** These metrics are guidelines. Actual performance is sensitive to **sample complexity**:
    *   **High Diversity:** Samples with a high number of unique species-level clades will consume more memory to maintain the k-mer hit counters.
    *   **Classification Depth:** Reads classified deeper in the tree (e.g., at species vs. family) require more taxonomic distance calculations, which can influence processing time.
*   **Infrastructure:** For large datasets (e.g., >10 GB per sample), use of high-memory nodes (>128 GB) is recommended for the ingestion phase.
