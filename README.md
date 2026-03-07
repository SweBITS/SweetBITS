# SweetBITS

Bioinformatics command-line tools for the SweBITS project (metagenomic shotgun sequencing of archived air filters).

See `GEMINI.md` for detailed technical specifications and architecture, and `USAGE.md` for practical examples.

## Installation

```bash
# Optional: install JolTax in editable mode
pip install -e /home/daniel/devel/JolTax

# Install SweetBITS in editable mode
pip install -e /home/daniel/devel/SweetBITS
```

## Commands Overview

SweetBITS provides several high-performance tools for processing Kraken 2 outputs:

- `gather-reports`: Merges multiple Kraken reports into a single, Polars-optimized Parquet file with full provenance metadata. Supports flexible SweBITS sample IDs (e.g., `Ki-2022_20_001`, `Lj_2013_1_142`, `Ki-2022-01-1`).
    - *Automatic Detection:* Handles both newer 8-column (with minimizers) and legacy 6-column Kraken reports automatically.
    - *Automatic Data Standard:* Automatically detects and adapts to SweBITS or Generic datasets based on input filenames.
- `table`: Generates wide-format abundance matrices. Supports three modes:
    - `taxon`: Direct taxonomic assignments.
    - `clade`: Cumulative clade counts (contains redundant counts).
    - `canonical`: **Canonical Remainders**. Essentially taxon mode but where reads between canonical ranks have been pushed up to the nearest canonical ancestor (NCA). Eliminates double-counting while conserving mass balance. Supports "non-canonical rank skipping" (Canonical Rank Read Standardization).
    - *Note:* `--exclude-samples` will issue a warning if an ID in your exclusion file is missing from the dataset.
- `annotate-table`: Transforms numeric abundance matrices into human-readable files. Automatically injects full taxonomic lineages from JolTax, calculates mean/median abundance, and sorts rows hierarchically (Superkingdom -> Species).
    - *External Metadata:* Seamlessly join any number of external TSV/CSV/Parquet files. The files **must** contain a `t_id` column. All other columns are automatically appended, and column collisions are safely resolved. The final table is ordered: `Taxonomy` -> `Metadata` -> `Summary Stats` -> `Samples`.
- `extract-reads`: Efficiently streams reads from Parquet files back into FASTQ.gz format with high throughput and a constant memory profile (OOM-safe). Supports TaxID and temporal filters.
- `inspect`: View provenance metadata, compression settings, and sorting information stored in SweetBITS Parquet files.

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
