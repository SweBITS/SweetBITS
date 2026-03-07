import polars as pl
from pathlib import Path
from typing import List, Optional
from sweetbits.utils import parse_sample_id
from sweetbits.metadata import get_standard_metadata, write_parquet_with_metadata

def parse_kraken_report(file_path: Path) -> pl.DataFrame:
    """
    Parses a single 8-column Kraken report file into a Polars DataFrame.
    """
    schema = {
        "column_1": pl.Float64, # Percentage
        "column_2": pl.UInt32,  # Clade reads
        "column_3": pl.UInt32,  # Taxon reads
        "column_4": pl.UInt64,  # MM total
        "column_5": pl.UInt32,  # MM unique
        "column_6": pl.String,  # Rank
        "column_7": pl.UInt32,  # TaxID
        "column_8": pl.String,  # Name
    }
    
    df = pl.read_csv(
        file_path,
        has_header=False,
        separator="\t",
        schema=schema,
        new_columns=["pct", "clade_reads", "taxon_reads", "mm_tot", "mm_uniq", "rank", "t_id", "name"]
    )
    
    return df.select(["clade_reads", "taxon_reads", "mm_tot", "mm_uniq", "t_id"])

def gather_reports_logic(
    input_dir: Path,
    output_file: Path,
    recursive: bool = True,
    include_pattern: str = "*.report"
):
    """
    Finds and merges Kraken report files into a single Parquet file.
    """
    search_path = "**/" + include_pattern if recursive else include_pattern
    report_files = list(input_dir.glob(search_path))
    
    if not report_files:
        raise FileNotFoundError(f"No files matching '{include_pattern}' found in {input_dir}")
        
    dfs = []
    for file_path in report_files:
        sample_id = file_path.name.split('.')[0]
        info = parse_sample_id(sample_id)
        df = parse_kraken_report(file_path)
        
        df = df.with_columns([
            pl.lit(sample_id).alias("sample_id"),
            pl.lit(info["year"]).cast(pl.UInt16).alias("year"),
            pl.lit(info["week"]).cast(pl.UInt8).alias("week"),
            pl.lit(str(file_path.relative_to(input_dir))).alias("source_file")
        ])
        dfs.append(df)
        
    merged_df = pl.concat(dfs)
    
    final_cols = [
        "sample_id", "year", "week", "t_id", 
        "clade_reads", "taxon_reads", "mm_tot", "mm_uniq", 
        "source_file"
    ]
    
    sorting = "year, week, sample_id, t_id"
    merged_df = merged_df.select(final_cols).sort(["year", "week", "sample_id", "t_id"])
    
    # Save with zstd compression and metadata
    metadata = get_standard_metadata(
        file_type="REPORT_PARQUET", 
        source_path=input_dir,
        compression="zstd (level 3)",
        sorting=sorting
    )
    write_parquet_with_metadata(
        merged_df, 
        output_file, 
        metadata, 
        compression="zstd", 
        compression_level=3
    )
