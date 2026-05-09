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

## 2. Grand Global K-mer Features (`kmer-global`)
This engine pools all k-mer classification data across any number of samples to create a "Grand Total" evidence profile for every species in the dataset.

#### A. Core Count Metrics
- **`grand_clade_kmers`**: Total k-mers classified to the target species or its descendants.
- **`grand_classified_kmers`**: Total k-mers that received *any* taxonomic assignment.
- **`grand_total_kmers`**: Absolute total of all k-mers (including unclassified).
- **`grand_lineage_kmers`**: Total classified k-mers hitting the taxonomic lineage (e.g., hitting the Genus but not the Species).
- **`grand_misclassified_kmers`**: K-mers classified outside the species AND outside its lineage (potential noise).

#### B. Evidence Ratios
- **`grand_clade_to_classified_kmer_ratio`**: Proportion of classified evidence that correctly hits the clade.
- **`grand_lineage_to_classified_kmer_ratio`**: Proportion of classified evidence hitting the lineage (high-level assignment).
- **`grand_misclassified_to_classified_kmer_ratio`**: Proportion of classified evidence hitting unrelated taxa.
- **`grand_supporting_to_misclassified_kmer_ratio`**: Ratio of "Good" hits (Clade + Lineage) to "Bad" hits (Misclassified).
- **`grand_clade_to_total_kmer_ratio`**: The global Kraken 2 Confidence Score for this species.

#### C. Taxonomic Distance & Depth (Weighted Stats)
*Metrics: `mean_`, `median_`, `cv_`, `p05_`, `p95_` for:*
- **`grand_misclassified_kmer_distance`**: The number of nodes in the tree between the assigned species and the unrelated k-mer hit. Large distances suggest egregious misclassifications.
- **`grand_misclassified_kmer_depth`**: The absolute depth in the tree of the taxa where misclassified k-mers hit.
- **`grand_misclassified_kmer_relative_lca_depth`**: Depth of the Lowest Common Ancestor (LCA) relative to the target species depth. Values near 1.0 mean the "noise" is taxonomically close to the target.
- **`grand_lineage_kmer_relative_depth`**: Where in the lineage k-mer hits are clustering (e.g., just above species vs. near root).

#### D. Top Hit Profiles
- **`grand_top_5_misclassified_kmer_names`**: Names of the top 5 unrelated taxonomic competitors.
- **`grand_top_5_misclassified_kmer_tax_ids`**: TaxIDs of the top 5 competitors.
- **`grand_top_5_misclassified_kmer_shares`**: Percentage of total misclassified k-mers held by each of the top 5 taxa.

---

### **Computational Complexity & Performance**
The generation of global k-mer features is a two-step process: **Ingestion** (parsing raw Kraken files) and **Extraction** (pooling data).

*   **Scaling:** Ingestion is the primary bottleneck. Based on the **Ljungbyhed dataset**, memory scales at **~4.8 GB per 1 GB** of compressed input, while runtime scales at **~3.5 minutes per 1 GB** (on 4 cores).
*   **Beyond File Size:** These metrics are guidelines. Actual performance is sensitive to **sample complexity**:
    *   **High Diversity:** Samples with a high number of unique species-level clades will consume more memory to maintain the k-mer hit counters.
    *   **Classification Depth:** Reads classified deeper in the tree (e.g., at species vs. family) require more taxonomic distance calculations, which can influence processing time.
*   **Infrastructure:** For large datasets (e.g., >10 GB per sample), use of high-memory nodes (>128 GB) is recommended for the ingestion phase.
