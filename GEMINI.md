# SweetBITS*
*: Name TBD

## Overview
This is the master instruction document for an AI agent assisting in the development of **SweetBITS**, a suite of bioinformatics command-line tools tightly knit to data processing in the SweBITS project (Swedish Biodiversity in Time and Space).

### The SweBITS Context
SweBITS pioneers metagenomic shotgun sequencing of archived air filters from two Swedish sites (Ljungbyhed and Kiruna). 
- **Core characteristic:** Each sample covers a unique week in a unique year. 
- **Read depth:** 150-250 million paired-end reads per sample.
- **Reference Publication:** https://www.nature.com/articles/s41467-025-67676-7

#### Datasets
- **Kiruna:** ~380 samples (1974-2014, mostly even-numbered years). Heterogeneous summer weeks. Sample ID format: `Ki-YYYY_WW_ZZZ` (where ZZZ is 1-3 digits).
- **Ljungbyhed:** ~520 samples (2013-2022). Weekly coverage. Sample ID format: `Lj-YYYY_WW_ZZZ` (where ZZZ is 1-3 digits).
- **Generalization:** Supports both `_` and `-` as separators between all components (e.g., `Ki-2022-01-001`).

#### Taxonomic Classification
Reads are classified using a custom fork of Kraken 2 against a 3TB custom database (mixed NCBI/GTDB taxonomy, where Bacteria/Archaea TaxIDs increment from 5,000,000). 
- **Kraken Reports:** We use an updated 8-column format (Percentage, Clade reads, Taxon reads, Total minimizers, Unique minimizers, Rank code, TaxID, Scientific name).
- **Kraken Read-by-Read:** Outputs 6 columns (Classified status, Seq ID, TaxID, Length in bp, Number of Minimizer Hit Groups (MHG), LCA k-mer mapping string).

---

## Data Dictionary

> **AI DIRECTIVE:** Always refer to these schemas when writing data manipulation, Polars transformations, or validation code.

### 1. `<KRAKEN_PARQUET>`
A parquet file representing a single sample's read-by-read data, sorted by `t_id`.
**Metadata:** Must include:
- `sweetbits_version`
- `file_type: KRAKEN_PARQUET`
- `execution_command`
- `creation_time`
- `compression`: Compression algorithm used.
- `sorting`: Column(s) used for sorting.
- `source_path_abs`: Absolute path to the original source file.

| Column | Type | Description |
| :--- | :--- | :--- |
| `sample_id` | String | The sample ID |
| `year` | UInt16 | The ISO year of the sample |
| `week` | UInt8 | The ISO week of the sample |
| `read_id` | String | The read ID |
| `r1_qual` | String | The read 1 quality string |
| `r2_qual` | String | The read 2 quality string |
| `r1_seq` | String | The read 1 sequence |
| `r2_seq` | String | The read 2 sequence |
| `r1_len` | UInt8 | The read 1 length |
| `r2_len` | UInt8 | The read 2 length |
| `total_len` | UInt16 | The combined read length (R1 + R2) |
| `t_id` | UInt32 | The classified TaxID |
| `mhg` | UInt8 | The number of minimizer hit groups (MHG) |
| `kmer_string` | String | The Kraken 2 k-mer string |
| `kmers_total` | UInt8 | The total number of k-mers in the read pair |
| `kmers_ambig` | UInt8 | The number of ambiguous k-mers in the read pair |
| `kmers_clade` | UInt8 | The number of k-mers classified to the clade |
| `kmers_lineage` | UInt8 | The number of k-mers classified to the lineage |
| `kmers_misclassified` | UInt8 | The number of k-mers classified outside clade/lineage |
| `clade_ratio` | Float32 | Ratio of clade k-mers to non-ambiguous k-mers |
| `lineage_ratio` | Float32 | Ratio of lineage k-mers to non-ambiguous k-mers |
| `misclassified_ratio` | Float32 | Ratio of misclassified k-mers to non-ambiguous k-mers |

### 2. `<REPORT_PARQUET>`
A single long-format parquet file containing merged, relevant counts from multiple report files.
Sorted by `year`, `week`, `sample_id`, and `t_id`. Compressed with `zstd`.
**Metadata:** Must include:
- `sweetbits_version`
- `file_type: REPORT_PARQUET`
- `execution_command`
- `creation_time`
- `compression`: Compression algorithm used.
- `sorting`: Column(s) used for sorting.
- `source_path_abs`: Absolute path to the input directory.

| Column | Type | Description |
| :--- | :--- | :--- |
| `sample_id` | String | The sample ID |
| `year` | UInt16 | The ISO year of the sample |
| `week` | UInt8 | The ISO week of the sample |
| `t_id` | UInt32 | The classified TaxID |
| `clade_reads` | UInt32 | Reads assigned to the clade rooted at this taxon |
| `taxon_reads` | UInt32 | Reads assigned directly to this taxon |
| `mm_tot` | UInt64 | Total minimizer matches (includes duplicates) |
| `mm_uniq` | UInt32 | Estimated distinct minimizer matches |
| `source_file` | String | Path to the original Kraken report file (relative to input) |

---

## Toolkit Command Specifications

### Data Conversion Tools

#### `convert_kraken`
Converts Kraken output into Polars-optimized Parquet files.
- **Inputs:** `<KRAKEN_FILE>` (read-by-read output), matched R1/R2 FASTQ files.
- **Outputs:** `<KRAKEN_PARQUET>`
> **AI DIRECTIVE:** Before implementing `convert_kraken`, prompt the user to finalize the data format, discuss how to specify FASTQ file inputs, handle the read-count discrepancies (human reads removed from FASTQs), optimize memory limits, define provenance/metadata logging, and agree on the final command name.

#### `gather_reports`
Merges multiple 8-column Kraken reports into a single Parquet file.
- **Inputs:** A directory containing report files.
- **Outputs:** `<REPORT_PARQUET>`
- **Arguments:**
  - `DIRECTORY`: Input directory to search for reports.
  - `--output FILE`: Path to the output Parquet file.
  - `--recursive / --no-recursive`: Search subdirectories (Default: True).
  - `--include GLOB`: Pattern to match report files (Default: `*.report`).
- **Implementation Details:**
  - Sort by `[year, week, sample_id, t_id]`.
  - Compress with `zstd` (level 3).
  - Extract `sample_id` from filename (base name before all extensions).
  - Validate `sample_id` using `parse_sample_id()`.
  - Include `source_file` column for provenance (relative path).
  - **Write global Parquet metadata** (version, command, timestamp, absolute input path, compression, sorting).

#### `prune_parquet` (Future)
Reduces columns in `<KRAKEN_PARQUET>` files (e.g., dropping k-mer strings after GBM feature calculation) to save space.

### Data Manipulation Tools

#### `table`
Outputs abundance tables with `t_id` as the index and samples (YYYY_WW) as columns.
- **Inputs:** `<REPORT_PARQUET>`
- **Arguments:**
  - `--mode`: `[taxon, clade, canonical]` (Default: `clade`)
  - `--output FILE`: Path to the output file (Supported: `.csv`, `.tsv`, `.parquet`). Format inferred from suffix.
  - `--taxonomy DIR`: JolTax cache directory (Required for `canonical` mode or `--clade` filtering).
- **Filters (Optional):**
  - `--exclude_samples FILE`: Text file, one ID per line. (Note: A warning is issued if any ID in this file is not found in the dataset).
  - `--min_observed INT`: Taxon must be in at least INT samples (default: 25).
  - `--min_reads INT`: Max value across samples must be >= INT (default: 50).
  - `--clade INT`: Output only taxa rooted at this TaxID.
- **Flags:**
  - `--keep_unclassified`: (Default: False).
- **Abundance Modes Explained:**
  - `taxon`: Raw `taxon_reads` from the Kraken report (reads assigned directly to this TaxID).
  - `clade`: Raw `clade_reads` from the Kraken report (cumulative reads for this taxon and all its descendants). **Caution:** This mode contains redundant counts across ranks.
  - `canonical`: **Canonical Remainders**. Essentially taxon mode but where reads between canonical ranks have been pushed up to the nearest canonical ancestor (NCA). This eliminates double-counting while conserving mass balance.
    - Uses `clade_reads` as input.
    - Identifies the Nearest Canonical Ancestor (NCA) for every node.
    - Subtracts the sum of all canonical child clades from the parent's clade count.
    - Corrects for "skipped ranks" and non-canonical assignments (e.g., `subgenus`).
    - The sum of all remainders in a sample exactly equals the total reads.

#### `clr`
Takes `<RAW_TABLE>` (from `table`) and calculates Centered Log Ratio using Bayesian multiplicative replacement for zeroes.
> **AI DIRECTIVE:** Before implementing, ask the user how to handle taxa with extremely high proportions of zeroes.

#### `extract_reads`
Streams `<KRAKEN_PARQUET>` to extract reads into FASTQ format with minimal memory footprint.
- **Inputs:** `<KRAKEN_PARQUET>` file or directory.
- **Arguments:**
  - `--taxonomy DIR`: JolTax cache directory (Required).
  - `--tax_id LIST`: Comma-separated TaxIDs to extract.
  - `--output-dir DIR`: Directory to save FASTQ files.
  - `--mode [clade, taxon]`: Extraction mode (Default: clade).
  - `--combine-samples`: If True, merges all samples into one file per TaxID.
- **Temporal Filters:**
  - `--year-start`, `--year-end` (Optional)
  - `--week-start`, `--week-end` (Optional)
  - Interval logic: `(year, week) >= (start) AND (year, week) <= (end)`.
- **Naming Convention:**
  - Single Sample: `{sample_id}_{mode}_{tax_id}_{ShortName}_R[1/2].fastq.gz`
  - Combined: `combined_{mode}_{tax_id}_{ShortName}_{RangeTag}_R[1/2].fastq.gz`
  - `ShortName`: 
    - >1 word: First two words, 3 letters each, PascalCase (e.g., "Homo sapiens" -> "HomSap").
    - 1 word: Use whole word.
  - `RangeTag`: e.g., `2013W50-to-2014W02` (Only for combined files when time filtering is active).

#### `annotate_table`
Amends `<RAW_TABLE>` with JolTax lineage metadata and outputs `<ANNOTATED_TABLE>`.
- **Implementation Note:** This tool is responsible for sorting the final output. Preferred strategies:
  1. Taxonomic hierarchy (Domain -> Species).
  2. DFS of the taxonomy tree guided by mean/median abundance.
> **AI DIRECTIVE:** Prompt the user to discuss integration of external metadata (GBIF TP/FP files, sibling reference features, assembly stats, Kraken database minimizers) before coding this tool.

#### `to_krona`
Generates Krona plots from abundance tables. Needs further discussion.

### Inspection Tools

#### `inspect`
Prints the global metadata stored in a SweetBITS-generated Parquet file.
- **Inputs:** `<PARQUET_FILE>`
- **Outputs:** Formatted summary of provenance metadata.

---

## Architectural & Development Rules

### Agent Workflow (TDD)
1. **Test-Driven Development:** Write `pytest` cases *before* implementing the logic.
2. **Documentation Sync:** After each tool implementation, evaluate if `README.md`, `USAGE.md`, and `GEMINI.md` require updates. Keep the AI and human context perfectly aligned.
3. **Clarification:** Whenever an ambiguity is resolved, update this `GEMINI.md` file immediately to solidify the decision.

### Generalization Requirement
- **Isolate SweBITS Logic:** Create a dedicated parsing utility module (e.g., `parse_sample_id()`) for the `YYYY_WW_ZZZ` format. Do not hardcode this logic across the tools. This ensures SweetBITS can be easily generalized for non-SweBITS datasets in the future.

### Code Constraints & Libraries
- **Polars Over Pandas:** Strict requirement. Use `LazyFrames` by default. Apply streaming. Downcast datatypes (especially for Parquet saves). Explicitly manage core counts. Use `Categorical` types where sensible.
- **CLI Framework:** Use `click`.
- **Taxonomy:** Use `JolTax` (>=v0.2.0).
- **Standard Libraries:** Use `pathlib` for paths, `logging` for output, `numpy` for vectorization.
- **Performance:** Stream reads, chunk large operations, and utilize multiprocessing (`--cores` exposed in CLI) where reasonable. Fit within standard workstation RAM limits.

### UX & Logging
- **Standard Output Header:** Every command must log: Start time, Toolkit Version, CWD, and the exact Invocation Command.
- **Parameter Logging:** Immediately after the header, log all parameter settings (including defaults) in a dimmed/subtle style (`bright_black`).
- **Standard Output Footer:** Print a summary statement (e.g., records processed, active samples, time elapsed).
- **Sanity Warnings:** Commands should issue warnings (in yellow) when parameter combinations are likely to produce unintended results (e.g., filtering out >50% of samples).
- **Progress:** Use `click.progressbar` or similar for long-running tasks.
- **Docs:** Google docstrings for code. Simple, explanatory README.md with usage examples. Avoid marketing superlatives.

### Input Data Strict Assumptions
- Paired-end data of max 2x150bp.
- K-mer size (k) must be at least 35 (ensures k-mer counts fit in UInt8).
- 8-column Kraken report files.
- 6-column Kraken read-by-read files (custom fork containing MHG).

### Roadmap
1. [x] Generate test data (Ljungbyhed sample, 100 reads, mock taxonomy).
2. [x] Implement `gather_reports` to merge Kraken reports.
3. [x] Implement `table` for abundance matrix generation.
4. [x] Implement `extract_reads` for FASTQ streaming.
5. [ ] Implement `annotate_table` for taxonomic annotation and sorting.
6. [ ] TBD...
