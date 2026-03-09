# SweetBITS

**THIS IS A WORK IN PROGRESS** - README.md not in a final state.

Bioinformatics command-line tools for the SweBITS project (metagenomic shotgun sequencing of archived air filters).

See `GEMINI.md` for detailed technical specifications and architecture, and `USAGE.md` for practical examples.

## Installation

```bash
# Optional: install JolTax in editable mode
pip install -e /home/daniel/devel/JolTax

# Install SweetBITS in editable mode
pip install -e /home/daniel/devel/SweetBITS
```

## Command Structure

SweetBITS organizes its tools into a logical workflow centered around bringing data *into* the ecosystem (`collect`) and getting usable artifacts *out* (`produce`).

```text
sweetbits
├── collect
│   └── kraken
│       ├── reports            <- Gathers multiple kraken .report files
│       └── classifications    <- Ingests kraken read-by-read output + FASTQ
│
├── produce
│   ├── reads                  <- Extracts reads back to FASTQ
│   └── table                  <- Generates abundance matrices
│
├── annotate                   <- Amends tables with JolTax metadata
└── inspect                    <- Prints Parquet metadata
```

## Commands Overview

SweetBITS provides several high-performance tools for processing Kraken 2 outputs. All tools that generate output files feature strict **overwrite protection**; use the `--overwrite` flag to replace existing files.

- `collect kraken reports`: Merges multiple Kraken reports into a single, Polars-optimized Parquet file with full provenance metadata (saved to a JSON companion file). Supports flexible SweBITS sample IDs (e.g., `Ki-2022_20_001`, `Lj_2013_1_142`, `Ki-2022-01-1`).
    - *Automatic Detection:* Handles both newer 8-column (with minimizers) and legacy 6-column Kraken reports automatically.
    - *Automatic Data Standard:* Automatically detects and adapts to SweBITS or Generic datasets based on input filenames.
- `produce table`: Generates wide-format abundance matrices. Supports three modes:
    - `taxon`: Direct taxonomic assignments.
    - `clade`: Cumulative clade counts (contains redundant counts).
    - `canonical`: **Canonical Remainders**. Essentially taxon mode but where reads between canonical ranks have been pushed up to the nearest canonical ancestor (NCA). Eliminates double-counting while conserving mass balance. Supports "non-canonical rank skipping" (Canonical Rank Read Standardization).
    - *Note:* `--exclude-samples` will issue a warning if an ID in your exclusion file is missing from the dataset.
- `annotate`: Transforms numeric abundance matrices into human-readable files. Automatically injects full taxonomic lineages from JolTax, calculates mean/median signal, and sorts rows hierarchically (Domain/Superkingdom -> Species).
    - *External Metadata:* Seamlessly join any number of external TSV/CSV/Parquet files. The files **must** contain a `t_id` column. All other columns are automatically appended, and column collisions are safely resolved. The final table is ordered: `Taxonomy` -> `Metadata` -> `Summary Stats` -> `Samples`.
- `collect kraken classifications`: Ingests massive Kraken and FASTQ files into heavily compressed, memory-mapped `<KRAKEN_PARQUET>` data lakes using an extremely fast, multiprocessed two-pointer streaming engine. If FASTQ files are omitted, it automatically creates a "Skinny Parquet" that drops sequence payloads to save significant disk space while retaining all taxonomic intelligence.
- `produce reads`: Efficiently streams reads from Parquet files back into FASTQ.gz format with high throughput and a constant memory profile (OOM-safe). Supports TaxID and temporal filters.
- `inspect`: View provenance metadata, compression settings, and sorting information stored in SweetBITS JSON companion files for any generated output.

## Shell Autocompletion

SweetBITS supports shell autocompletion. To enable it for Bash, add this to your `~/.bashrc`:

```bash
eval "$(_SWEETBITS_COMPLETE=bash_source sweetbits)"
```

### For Conda Users
If you use Conda, ensure autocompletion only loads when the environment is active by using an activation script:

```bash
conda activate your_env_name
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'eval "$(_SWEETBITS_COMPLETE=bash_source sweetbits)"' > $CONDA_PREFIX/etc/conda/activate.d/sweetbits_completion.sh
```
