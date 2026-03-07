import polars as pl
import logging
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from joltax import JolTree
from sweetbits.utils import parse_sample_id
from sweetbits.metadata import get_standard_metadata, write_parquet_with_metadata

logger = logging.getLogger(__name__)

def generate_table_logic(
    input_parquet: Path,
    output_file: Path,
    mode: str = "taxon",
    taxonomy_dir: Optional[Path] = None,
    exclude_samples: Optional[Path] = None,
    min_observed: int = 25,
    min_reads: int = 50,
    clade_filter: Optional[int] = None,
    keep_unclassified: bool = False
) -> Dict[str, Any]:
    """
    Generates an abundance table from a merged REPORT_PARQUET file.
    Returns a summary dictionary of the operation.
    """
    # 1. Load Data
    df = pl.read_parquet(input_parquet)
    
    total_samples_in_file = df["sample_id"].n_unique()
    
    # 2. Filtering Samples
    if exclude_samples:
        with open(exclude_samples, "r") as f:
            excluded_ids = [line.strip() for line in f if line.strip()]
        df = df.filter(~pl.col("sample_id").is_in(excluded_ids))
        
    active_samples = df["sample_id"].n_unique()
    
    # Warning for high min_observed
    if min_observed > (active_samples / 2) and active_samples > 0:
        import click
        click.secho(
            f"Warning: --min-observed ({min_observed}) is more than 50% of active samples ({active_samples}). "
            "This may result in an empty or very sparse table.", 
            fg="yellow", err=True
        )

    # 3. Mode Handling & Clade Filtering (Requires JolTax)
    tree = None
    if mode in ["clade", "canonical"] or clade_filter is not None:
        if not taxonomy_dir:
            raise ValueError(f"Taxonomy directory is required for mode '{mode}' or clade filtering.")
        tree = JolTree.load(str(taxonomy_dir))
        
    # 3a. Clade Filter
    if clade_filter is not None:
        clade_taxids = tree.get_clade(clade_filter)
        df = df.filter(pl.col("t_id").is_in(clade_taxids))
        
    # 3b. Unclassified Handling
    if not keep_unclassified:
        df = df.filter(pl.col("t_id") != 0)
        
    # 4. Aggregation based on Mode
    if mode == "taxon":
        pivot_col = "taxon_reads"
    elif mode == "clade":
        pivot_col = "clade_reads"
    elif mode == "canonical":
        unique_tids = df["t_id"].unique().to_numpy()
        annotation = tree.annotate(unique_tids, strict=False)
        df = df.join(annotation.select(["t_id", "t_species"]), on="t_id")
        df = df.with_columns(pl.col("t_species").fill_null(pl.col("t_id").cast(pl.String)))
        pivot_col = "taxon_reads"
    
    # 5. Pivot to wide format
    df = df.with_columns(
        (pl.col("year").cast(pl.String) + "_" + pl.col("week").cast(pl.String).str.pad_start(2, "0")).alias("period")
    )
    
    table = df.pivot(
        values=pivot_col,
        index="t_id",
        on="period",
        aggregate_function="sum"
    ).fill_null(0)
    
    # 6. Apply Filters (min_observed, min_reads)
    sample_cols = [c for c in table.columns if c != "t_id"]
    
    if not sample_cols:
        table = pl.DataFrame(schema={"t_id": pl.UInt32})
    else:
        if min_observed > 0:
            obs_count = table.select([
                pl.sum_horizontal([pl.col(c) > 0 for c in sample_cols]).alias("count")
            ])["count"]
            table = table.filter(obs_count >= min_observed)
            
        if min_reads > 0:
            max_reads = table.select([
                pl.max_horizontal(sample_cols).alias("max_val")
            ])["max_val"]
            table = table.filter(max_reads >= min_reads)
        
    # 7. Output based on extension
    ext = output_file.suffix.lower()
    if ext == ".parquet":
        metadata = get_standard_metadata(
            file_type="RAW_TABLE", 
            source_path=input_parquet,
            compression="Uncompressed (Snappy default)",
            sorting="None"
        )
        write_parquet_with_metadata(table, output_file, metadata)
    elif ext == ".csv":
        table.write_csv(output_file)
    elif ext == ".tsv":
        table.write_csv(output_file, separator="\t")
    else:
        raise ValueError(f"Unsupported output format: {ext}")
        
    return {
        "active_samples": active_samples,
        "rows_output": table.height,
        "output_file": str(output_file)
    }
