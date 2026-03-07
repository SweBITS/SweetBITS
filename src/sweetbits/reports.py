"""
sweetbits.reports
Logic for parsing and merging Kraken 2 report files.
"""

import polars as pl
from pathlib import Path
from typing import List, Optional
from sweetbits.utils import parse_sample_id
from sweetbits.metadata import get_standard_metadata, write_parquet_with_metadata

def parse_kraken_report(file_path: Path) -> pl.DataFrame:
    """
    Parses a single 8-column Kraken report file into a Polars DataFrame.

    This function expects the updated 8-column Kraken 2 format which includes
    estimated unique minimizer counts. It enforces a strict schema for 
    memory efficiency and performance.

    Args:
        file_path: Path to the raw Kraken report text file.

    Returns:
        A Polars DataFrame containing the columns: 
        ['clade_reads', 'taxon_reads', 'mm_tot', 'mm_uniq', 't_id'].
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
) -> None:
    """
    Finds and merges multiple Kraken report files into a single long-format Parquet file.

    The process involves:
    1. Identifying files based on the include_pattern.
    2. Detecting the data standard (SWEBITS vs GENERIC) based on filenames.
    3. Normalizing temporal metadata if SWEBITS standard is detected.
    4. Vertically stacking all reports.
    5. Sorting for optimized downstream access.
    6. Injecting Arrow-level provenance metadata including 'data_standard'.

    Args:
        input_dir: Directory containing Kraken report files.
        output_file: Path where the merged Parquet file will be saved.
        recursive: Whether to search subdirectories. Defaults to True.
        include_pattern: Glob pattern to match report files. Defaults to "*.report".

    Raises:
        FileNotFoundError: If no files matching the pattern are found.
    """
    search_path = "**/" + include_pattern if recursive else include_pattern
    report_files = list(input_dir.glob(search_path))
    
    if not report_files:
        raise FileNotFoundError(f"No files matching '{include_pattern}' found in {input_dir}")
        
    # 1. Determine Data Standard
    is_swebits = True
    parsed_metadata = []
    
    for f in report_files:
        sample_id = f.name.split('.')[0]
        try:
            info = parse_sample_id(sample_id)
            parsed_metadata.append(info)
        except ValueError:
            is_swebits = False
            break
            
    data_standard = "SWEBITS" if is_swebits else "GENERIC"
    
    # 2. Process Files
    dfs = []
    for i, file_path in enumerate(report_files):
        sample_id = file_path.name.split('.')[0]
        df = parse_kraken_report(file_path)
        
        cols = {
            "sample_id": pl.lit(sample_id),
            "source_file": pl.lit(str(file_path.relative_to(input_dir)))
        }
        
        if is_swebits:
            info = parsed_metadata[i]
            cols["year"] = pl.lit(info["year"]).cast(pl.UInt16)
            cols["week"] = pl.lit(info["week"]).cast(pl.UInt8)
            
        df = df.with_columns(**cols)
        dfs.append(df)
        
    merged_df = pl.concat(dfs)
    
    # 3. Finalize Schema and Sort
    if is_swebits:
        final_cols = [
            "sample_id", "year", "week", "t_id", 
            "clade_reads", "taxon_reads", "mm_tot", "mm_uniq", 
            "source_file"
        ]
        sorting = "year, week, sample_id, t_id"
        merged_df = merged_df.select(final_cols).sort(["year", "week", "sample_id", "t_id"])
    else:
        final_cols = [
            "sample_id", "t_id", 
            "clade_reads", "taxon_reads", "mm_tot", "mm_uniq", 
            "source_file"
        ]
        sorting = "sample_id, t_id"
        merged_df = merged_df.select(final_cols).sort(["sample_id", "t_id"])
    
    # 4. Save with Metadata
    metadata = get_standard_metadata(
        file_type="REPORT_PARQUET", 
        source_path=input_dir,
        compression="zstd (level 3)",
        sorting=sorting,
        data_standard=data_standard
    )
    
    write_parquet_with_metadata(
        merged_df, 
        output_file, 
        metadata, 
        compression="zstd", 
        compression_level=3
    )
