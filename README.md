# SweetBITS

Bioinformatics command-line tools for the SweBITS project (metagenomic shotgun sequencing of archived air filters).

See `GEMINI.md` for detailed technical specifications and architecture.

## Installation

```bash
# Optional: install JolTax in editable mode
pip install -e /home/daniel/devel/JolTax

# Install SweetBITS in editable mode
pip install -e /home/daniel/devel/SweetBITS
```

## Commands Overview

SweetBITS provides several high-performance tools for processing Kraken 2 outputs:

- `gather-reports`: Merges multiple 8-column Kraken reports into a single, Polars-optimized Parquet file with full provenance metadata.
- `table`: Generates wide-format abundance matrices. Supports three modes:
    - `taxon`: Direct taxonomic assignments.
    - `clade`: Cumulative clade counts (contains redundant counts).
    - `canonical`: **Canonical Remainders**. Eliminates double-counting by attributing reads only to the most specific standard rank (Species, Genus, etc.). Sum of remainders conserved mass balance.
- `extract-reads`: Efficiently streams reads from Parquet files back into FASTQ.gz format based on TaxID and temporal filters (Year/Week).
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
