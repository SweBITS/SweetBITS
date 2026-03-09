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

Create a wide-format matrix from merged reports.

```bash
# Default (Clade mode, SWEBITS period grouping)
sweetbits produce table merged_reports.parquet --output abundance_table.tsv

# Taxon mode with specific filtering
sweetbits produce table merged_reports.parquet \
    --output taxon_table.csv \
    --mode taxon \
    --min-observed 50 \
    --min-reads 100

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
    --keep-composition
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
  3. `median_signal` and `mean_signal` (dynamically calculated)
  4. The raw sample abundance matrix

```bash
# Basic taxonomy annotation and hierarchical sorting
sweetbits annotate canonical_table.tsv \
    --taxonomy /path/to/joltax_cache \
    --output annotated_canonical.tsv

# Join multiple external metadata files (e.g., GBIF flags, Kraken stats)
sweetbits annotate abundance_table.parquet \
    --taxonomy /path/to/joltax_cache \
    --metadata gbif_status.csv \
    --metadata assembly_metrics.tsv \
    --output highly_annotated.csv
```

## 4. Extracting Reads (`produce reads`)

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
```

