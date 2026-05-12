# SweetBITS Usage Guide

This guide provides practical examples for the SweetBITS command-line tools.

**Overwrite Protection:** All tools that generate output files will refuse to overwrite an existing file unless the `--overwrite` flag is provided.

## 1. Merging Kraken Reports (`collect kraken reports`)

Merge multiple `.report` files from a directory into a single optimized Parquet file.

```bash
# Basic usage
sweetbits collect kraken reports /path/to/reports --output merged_reports.parquet

# Overwrite an existing output file
sweetbits collect kraken reports /path/to/reports --output existing.parquet --overwrite

# Recursive search with specific pattern
sweetbits collect kraken reports /path/to/data --output results.parquet --recursive --include "*.kraken.report"
```

## 2. Generating Abundance Tables (`produce table`)

Create a wide-format matrix from merged reports. All modes now require the JolTax taxonomy for dynamic clade calculation and recursive filtering.

```bash
# Default (Clade mode, SWEBITS period grouping)
sweetbits produce table merged_reports.parquet \
    --taxonomy /path/to/joltax_cache \
    --output abundance_table.tsv

# Dry-run: Preview filtering retention statistics without saving
sweetbits produce table merged_reports.parquet \
    --mode canonical \
    --taxonomy /path/to/joltax_cache \
    --min-observed 25 \
    --min-reads 50 \
    --dry-run

# Taxon mode with specific filtering
sweetbits produce table merged_reports.parquet \
    --output taxon_table.csv \
    --mode taxon \
    --taxonomy /path/to/joltax_cache \
    --min-observed 50 \
    --min-reads 100
```
# Canonical Remainders (Requires JolTax)
sweetbits produce table merged_reports.parquet \
    --output canonical_table.tsv \
    --mode canonical \
    --taxonomy /path/to/joltax_cache

# Filter for a specific clade (e.g., Bacteria = 2)
sweetbits produce table merged_reports.parquet \
    --output bacteria_only.tsv \
    --taxonomy /path/to/joltax_cache \
    --clade 2

# Output relative proportions instead of raw counts
sweetbits produce table merged_reports.parquet \
    --output proportions_table.tsv \
    --mode taxon \
    --proportions

# Calculate global proportions of Bacteria by keeping filtered reads in the total
sweetbits produce table merged_reports.parquet \
    --output global_bacteria_proportions.tsv \
    --mode canonical \
    --taxonomy /path/to/joltax_cache \
    --clade 2 \
    --proportions \
    --keep-filtered
```

## 3. Annotating Tables (`annotate`)

Transform numeric abundance matrices into human-readable files sorted by taxonomy, and integrate external metadata.

### How External Metadata Works
You can join any number of external metadata files (CSV, TSV, or Parquet) to your abundance table. 
- **Requirement:** Every metadata file **MUST** contain a `t_id` column. This is used as the key for the left-join.
- **What gets added:** Every column from the metadata file (except `t_id`) will be appended to the output table.
- **Collisions:** If a metadata file contains a column name that already exists in your table, SweetBITS will automatically append the filename to the column (e.g., `status` becomes `status_gbif_flags`) and issue a warning.
- **Column Order:** The final output is strictly ordered to maximize readability:
  1. `t_id` and all taxonomic ranks (`t_scientific_name`, `t_phylum`, etc.)
  2. All external metadata columns (in the order the files were provided)
  3. `sig_avg` and `sig_med` (if --add-stats is used)
  4. The raw input data columns

```bash
# Basic taxonomy annotation and hierarchical sorting (alphabetical)
sweetbits annotate canonical_table.tsv \
    --taxonomy /path/to/joltax_cache \
    --output annotated_canonical.tsv

# Abundance-weighted DFS sorting (related organisms cluster together, most abundant first)
sweetbits annotate taxon_table.csv \
    --taxonomy /path/to/joltax_cache \
    --sort-order dfs \
    --output annotated_dfs.csv

# Join multiple external metadata files (e.g., GBIF flags, Kraken stats)
sweetbits annotate abundance_table.parquet \
    --taxonomy /path/to/joltax_cache \
    --metadata gbif_status.csv \
    --metadata assembly_metrics.tsv \
    --output highly_annotated.csv
```

## 4. Calculating Features (`produce feature`)

Generate taxonomic validation metrics and other complex features.

### Minimizer Correlations (`uniq-minimizer-corr`)
This tool calculates the statistical reliability of taxonomic assignments by correlating observed unique minimizer coverage against a probabilistic model. It generates a "Validation Metadata" table that can be joined to any abundance matrix.

```bash
# Basic usage
sweetbits produce feature uniq-minimizer-corr merged_reports.parquet \
    --inspect /path/to/kraken_inspect.csv \
    --taxonomy /path/to/joltax_cache \
    --output minimizer_validation.csv

# Exclude bad samples and overwrite existing output
sweetbits produce feature uniq-minimizer-corr merged_reports.parquet \
    --inspect kraken_inspect.csv \
    --taxonomy /path/to/joltax_cache \
    --bad-samples bad_samples.txt \
    --output minimizer_validation.parquet \
    --overwrite
```

### Global K-mer Features (`kmer-global`)
Calculates globally aggregated k-mer features by pooling data from multiple samples. This is a two-step process: Ingestion followed by Feature Extraction.

#### Step 1: Ingest K-mers
```bash
# Process a single kraken file
sweetbits collect kraken kmers Ki-2022_01_001.kraken \
    --taxonomy /path/to/joltax_cache \
    --output-dir ./kmer_aggregation
```

#### Step 2: Extract Global Features
```bash
# Pool all ingested samples to create global profiles
sweetbits produce feature kmer-global "./kmer_aggregation/*.kmers.parquet" \
    --taxonomy /path/to/joltax_cache \
    --output kmers_global_features.csv
```

### Per-Sample K-mer Features & Stability
For temporal analysis and machine learning, you can extract long-format features and calculate inter-sample stability.

#### Step 1: Extract Per-Sample Features
```bash
# Generate a temporal Parquet dataset of k-mer evidence
sweetbits produce feature kmer-sample "./kmer_aggregation/*.kmers.parquet" \
    --taxonomy /path/to/joltax_cache \
    --output kmers_sample_features.parquet
```

#### Step 2: Calculate Stability Meta-Features
```bash
# Calculate variance and stability scores from the long-format output
sweetbits produce feature kmer-stability kmers_sample_features.parquet \
    --output kmers_stability_features.csv
```

### Scalable "Divide and Conquer" Workflow
For massive datasets (500+ samples), processing all k-mer features in a single pass may exceed workstation RAM. In these cases, use the chunked workflow:

#### Phase 1: Generate Intermediate Chunks
Run `kmer-sample` on individual files (easily parallelizable in Snakemake):
```bash
sweetbits produce feature kmer-sample sample_001.kmers.parquet \
    --taxonomy /path/to/joltax_cache \
    --output sample_001.chunk.parquet
```

#### Phase 2: Merge Chunks
Use the high-speed collection tool to unify the temporal dataset:
```bash
sweetbits collect feature-chunks "./chunks/*.chunk.parquet" \
    --output sample_kmer_features.parquet
```

#### Phase 3: Calculate Stability
Run the stability engine on the merged file as normal.

### Full Validation Workflow
A typical SweetBITS workflow for creating a high-confidence abundance table:

1. **Merge Reports**: 
   `sweetbits collect kraken reports /data --output reports.parquet`
2. **Generate Table**: 
   `sweetbits produce table reports.parquet --mode canonical --output raw_table.tsv`
3. **Calculate Validation**: 
   `sweetbits produce feature uniq-minimizer-corr reports.parquet --inspect kraken_inspect.csv --output validation.csv`
4. **Annotate & Merge**: 
   `sweetbits annotate raw_table.tsv --metadata validation.csv --output validated_table.tsv`

## 5. Extracting Reads (`produce reads`)

Stream reads from annotated Parquet files back to FASTQ.gz format.

```bash
# Extract specific TaxIDs
sweetbits produce reads /path/to/parquet_dir \
    --taxonomy /path/to/joltax_cache \
    --tax-id "9606,10090" \
    --output-dir ./extracted_reads

# Overwrite existing files in the output directory
sweetbits produce reads /path/to/parquet_dir \
    --taxonomy /path/to/joltax_cache \
    --tax-id "9606" \
    --output-dir ./existing_dir \
    --overwrite

# Combine all samples into one file per TaxID with temporal filtering
sweetbits produce reads /path/to/data \
    --taxonomy /path/to/joltax_cache \
    --tax-id "2" \
    --combine-samples \
    --year-start 2020 --year-end 2022 \
    --output-dir ./bacteria_reads
```

## 5. Ingestion (`collect kraken classifications`)

Convert raw Kraken and FASTQ files into high-performance, compressed Parquet data lakes.

```bash
# Basic ingestion (Fat Parquet)
sweetbits collect kraken classifications sample.kraken \
    --r1 sample_R1.fastq.gz \
    --r2 sample_R2.fastq.gz \
    --output sample.kraken.parquet

# High-performance Skinny Parquet (omit FASTQ files to save significant disk space)
# Highly recommended for standard workflows.
sweetbits collect kraken classifications sample.kraken \
    --cores 8 \
    --output sample_skinny.kraken.parquet
```

## 6. Inspecting Metadata (`inspect`)

View the provenance and configuration of any SweetBITS generated file via its JSON companion metadata.

```bash
sweetbits inspect merged_reports.parquet
sweetbits inspect abundance_table.csv.json

## 7. Performance & Resource Planning

Based on empirical benchmarks from the **Ljungbyhed dataset** (516 samples, ranging from 0.84 GB to 25.21 GB per compressed `.kraken2.gz` file), we have identified the following scaling patterns for the `collect kraken kmers` tool (using 4 cores):

| Metric | Scaling Factor (per 1 GB decimal) | Typical Range |
| :--- | :--- | :--- |
| **Memory (RSS)** | **~4.8 GB** (+ 3.5 GB baseline) | 10–130 GB |
| **Runtime** | **~3.5 minutes** | 15–90 minutes |

### **Guideline Note**
These numbers should be treated as **guidelines** rather than absolute limits. Resource usage is driven by more than just file size; it is also heavily influenced by:
*   **Sample Complexity:** High taxonomic diversity (many unique species) increases memory pressure during k-mer aggregation.
*   **Classification Rate:** Samples with a higher proportion of classified reads will require more processing time and memory than "empty" or poorly classified samples.
*   **I/O Performance:** The speed of the underlying storage system can significantly impact runtime, especially when streaming large gzipped files.

**Recommendation:** For stable execution, allocate at least **6 GB of RAM per 1 GB of compressed input**.
```

