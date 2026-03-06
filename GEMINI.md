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
- **Kiruna:** ~380 samples (1974-2014, mostly even-numbered years). Heterogeneous summer weeks. Sample ID format: `Ki-YYYY_WW_ZZZ`.
- **Ljungbyhed:** ~520 samples (2013-2022). Weekly coverage. Sample ID format: `Lj-YYYY_WW_ZZZ`.

#### Taxonomic Classification
Reads are classified using a custom fork of Kraken 2 against a 3TB custom database (mixed NCBI/GTDB taxonomy, where Bacteria/Archaea TaxIDs increment from 5,000,000). 
- **Kraken Reports:** We use an updated 8-column format (Percentage, Clade reads, Taxon reads, Total minimizers, Unique minimizers, Rank code, TaxID, Scientific name).
- **Kraken Read-by-Read:** Outputs 6 columns (Classified status, Seq ID, TaxID, Length in bp, Number of Minimizer Hit Groups (MHG), LCA k-mer mapping string).

---

## Data Dictionary

> **AI DIRECTIVE:** Always refer to these schemas when writing data manipulation, Polars transformations, or validation code.

### 1. `<KRAKEN_PARQUET>`
A parquet file representing a single sample's read-by-read data, sorted by `t_id`.
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
- **Inputs:** Multiple `<REPORT_FILE>`s or a directory.
- **Outputs:** `<REPORT_PARQUET>`
> **AI DIRECTIVE:** Before implementing, prompt the user to discuss preferred Parquet sorting strategies, compression algorithms, directory input UX, and provenance metadata tracking.

#### `prune_parquet` (Future)
Reduces columns in `<KRAKEN_PARQUET>` files (e.g., dropping k-mer strings after GBM feature calculation) to save space.

### Data Manipulation Tools

#### `table`
Outputs abundance tables with `t_id` as the index and samples (YYYY_WW) as columns.
- **Inputs:** `<REPORT_PARQUET>`
- **Arguments:**
  - `--mode`: `[taxon, clade, canonical]`
  - Output file/folder
  - JolTax cache directory (Required for `--clade` or `canonical`)
- **Filters (Optional):**
  - `--exclude_samples FILE`: Text file, one ID per line.
  - `--min_observed INT`: Taxon must be in at least INT samples (default: 25).
  - `--min_reads INT`: Max value across samples must be >= INT (default: 50).
  - `--clade INT`: Output only taxa rooted at this TaxID.
- **Flags:**
  - `--keep_unclassified`: (Default: False).
  - `--parquet_output`: (Default: False, outputs CSV otherwise).
> **AI DIRECTIVE:** Before implementing, prompt the user to discuss output sorting preferences and whether the UX should accept an output file path or a directory path.

#### `clr`
Takes `<RAW_TABLE>` (from `table`) and calculates Centered Log Ratio using Bayesian multiplicative replacement for zeroes.
> **AI DIRECTIVE:** Before implementing, ask the user how to handle taxa with extremely high proportions of zeroes.

#### `extract_reads`
Streams `<KRAKEN_PARQUET>` to extract reads into FASTQ format with minimal memory footprint.
- **Required:** `--input FILE`, `--taxonomy DIR` (JolTax), `--tax_id LIST` (comma-separated Ints).
- **Optional:** `--year LIST`, `--week LIST`, `--combine_output` (Default False: sample-specific files), `--mode [clade, taxon]` (Default: clade).

#### `annotate_table`
Amends `<RAW_TABLE>` with JolTax lineage metadata and outputs `<ANNOTATED_TABLE>`.
> **AI DIRECTIVE:** Prompt the user to discuss integration of external metadata (GBIF TP/FP files, sibling reference features, assembly stats, Kraken database minimizers) before coding this tool.

#### `to_krona`
Generates Krona plots from abundance tables. Needs further discussion.

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
- **Standard Output Footer:** Print a summary statement (e.g., rows processed, time elapsed, memory used).
- **Progress:** Use `click.progressbar` or similar for long-running tasks.
- **Docs:** Google docstrings for code. Simple, explanatory USAGE.md with a machine-readable API section. Avoid marketing superlatives.

### Input Data Strict Assumptions
- Paired-end data of max 2x150bp.
- 8-column Kraken report files.
- 6-column Kraken read-by-read files (custom fork containing MHG).

### Roadmap
1. Generate test data (Ljungbyhed sample, 100 reads, varying qual/len).
2. TBD...