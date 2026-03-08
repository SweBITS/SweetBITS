"""
sweetbits.annotate
Logic for annotating abundance tables with taxonomy and external metadata.
"""

import polars as pl
import logging
import click
from pathlib import Path
from typing import Optional, List, Dict, Any
from joltax import JolTree

logger = logging.getLogger(__name__)

def annotate_table_logic(
    input_table: Path,
    taxonomy_dir: Path,
    output_file: Path,
    metadata_files: Optional[List[Path]] = None
) -> Dict[str, Any]:
    """
    Annotates a raw abundance table with taxonomic lineages and external metadata.

    Loads a `<RAW_TABLE>`, queries JolTax for full taxonomic lineages (prefixed with 't_'),
    calculates summary statistics (mean/median), and joins any user-provided metadata files.
    Finally, sorts the table hierarchically and structures the columns.

    Args:
        input_table    : Path to the raw abundance table (Parquet, CSV, TSV).
        taxonomy_dir   : Path to the JolTax cache directory.
        output_file    : Path where the annotated table will be saved.
        metadata_files : Optional list of paths to external metadata files to join.

    Returns:
        A dictionary containing processing statistics:
        - 'taxa_processed' : Total number of taxa in the input table.
        - 'output_file'    : Path to the saved annotated table.

    Raises:
        ValueError : If a metadata file lacks a 't_id' column or if an unsupported format is used.
    """
    if metadata_files is None:
        metadata_files = []

    # 1. Load Base Table
    ext = input_table.suffix.lower()
    if ext == ".parquet":
        df = pl.read_parquet(input_table)
    elif ext == ".tsv":
        df = pl.read_csv(input_table, separator="\t")
    else:
        df = pl.read_csv(input_table)

    if "t_id" not in df.columns:
        raise ValueError(f"Input table {input_table.name} must contain a 't_id' column.")

    sample_cols = [c for c in df.columns if c != "t_id"]
    base_tids = df["t_id"].to_list()
    base_tids_set = set(base_tids)
    num_taxa = len(base_tids)

    # 2. JolTax Annotation
    tree = JolTree.load(str(taxonomy_dir))
    
    # We enforce strict=True because the JolTax cache MUST match the Kraken database
    # used to generate the abundance table. Missing TaxIDs indicate a critical config error.
    tax_df = tree.annotate(base_tids, strict=True)
    click.secho(f"Annotated {num_taxa}/{num_taxa} taxa using JolTax taxonomy", fg="green", err=True)
    
    tax_cols = tax_df.columns
    df = df.join(tax_df, on="t_id", how="left")

    # 3. Metadata Loop
    metadata_cols = []
    for m_path in metadata_files:
        m_ext = m_path.suffix.lower()
        if m_ext == ".parquet":
            m_df = pl.read_parquet(m_path)
        elif m_ext in [".tsv", ".txt"]:
            m_df = pl.read_csv(m_path, separator="\t")
        elif m_ext == ".csv":
            m_df = pl.read_csv(m_path)
        else:
            raise ValueError(f"Unsupported metadata file format '{m_ext}' for {m_path.name}. Supported formats are .csv, .tsv, .txt, .parquet")

        if "t_id" not in m_df.columns:
            raise ValueError(f"Metadata file {m_path.name} must contain a 't_id' column.")
            
        if len(m_df.columns) == 1:
            click.secho(
                f"Warning: Metadata file {m_path.name} only contains 1 column ('{m_df.columns[0]}'). "
                f"This might indicate a separator mismatch or an empty metadata file.", 
                fg="yellow", err=True
            )

        m_tids = set(m_df["t_id"].to_list())
        intersect = len(base_tids_set.intersection(m_tids))
        click.secho(f"Annotated {intersect}/{num_taxa} taxa using {m_path.name}", fg="green", err=True)

        current_cols = set(df.columns)
        rename_map = {}
        new_m_cols = []

        for c in m_df.columns:
            if c == "t_id":
                continue
            if c in current_cols:
                new_name = f"{c}_{m_path.stem}"
                rename_map[c] = new_name
                new_m_cols.append(new_name)
                click.secho(
                    f"Warning: Column collision for '{c}' from {m_path.name}. Renamed to '{new_name}'.", 
                    fg="yellow", err=True
                )
            else:
                new_m_cols.append(c)

        if rename_map:
            m_df = m_df.rename(rename_map)

        df = df.join(m_df, on="t_id", how="left")
        metadata_cols.extend(new_m_cols)

    # 4. Summary Statistics
    if sample_cols:
        df = df.with_columns([
            pl.concat_list(sample_cols).list.median().alias("median_abundance"),
            pl.mean_horizontal(sample_cols).alias("mean_abundance")
        ])
    else:
        df = df.with_columns([
            pl.lit(0.0).alias("median_abundance"),
            pl.lit(0.0).alias("mean_abundance")
        ])

    # 5. Sorting (Hierarchical Taxonomy -> t_id)
    sort_cols_target = [
        "t_superkingdom", "t_phylum", "t_class", "t_order", 
        "t_family", "t_genus", "t_species", "t_id"
    ]
    actual_sort_cols = [c for c in sort_cols_target if c in df.columns]
    df = df.sort(actual_sort_cols, nulls_last=True)

    # 6. Column Ordering
    ordered_cols = tax_cols + metadata_cols + ["median_abundance", "mean_abundance"] + sample_cols
    # Ensure t_id is first if it isn't already (though JolTax annotate puts it first)
    if ordered_cols[0] != "t_id":
        ordered_cols.remove("t_id")
        ordered_cols.insert(0, "t_id")
        
    # Deduplicate while preserving order, just in case
    seen = set()
    final_cols = []
    for c in ordered_cols:
        if c not in seen:
            final_cols.append(c)
            seen.add(c)
            
    df = df.select(final_cols)

    # 7. Output Generation
    out_ext = output_file.suffix.lower()
    if out_ext == ".parquet":
        # Keep standard Parquet writing for tables
        df.write_parquet(output_file, compression="zstd", compression_level=3)
    elif out_ext == ".tsv":
        df.write_csv(output_file, separator="\t")
    else:
        df.write_csv(output_file)

    return {
        "taxa_processed": num_taxa,
        "output_file": str(output_file)
    }
