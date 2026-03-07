"""
sweetbits.reports
Logic for parsing and merging Kraken 2 report files.
"""

import polars as pl
from pathlib import Path
from typing import List, Optional, Tuple
from sweetbits.utils import parse_sample_id
from sweetbits.metadata import get_standard_metadata, write_parquet_with_metadata

def detect_report_format(file_path: Path) -> str:
    """
    Sniffs the first line of a Kraken report to determine its format.
    
    Returns:
        'HYPERLOGLOG' if 8 columns, 'LEGACY' if 6 columns.
    
    Raises:
        ValueError: If the column count is neither 6 nor 8.
    """
    with open(file_path, "r") as f:
        first_line = f.readline()
        if not first_line:
            raise ValueError(f"Report file is empty: {file_path}")
        cols = len(first_line.split('\t'))
        
    if cols == 8:
        return "HYPERLOGLOG"
    elif cols == 6:
        return "LEGACY"
    else:
        raise ValueError(f"Unsupported Kraken report format ({cols} columns) in {file_path.name}")

def parse_kraken_report(file_path: Path, report_format: str) -> pl.DataFrame:
    """
    Parses a Kraken report file based on the detected format.
    """
    if report_format == "HYPERLOGLOG":
        schema = {
            "column_1": pl.Float64, "column_2": pl.UInt32, "column_3": pl.UInt32,
            "column_4": pl.UInt64, "column_5": pl.UInt32, "column_6": pl.String,
            "column_7": pl.UInt32, "column_8": pl.String,
        }
        new_names = ["pct", "clade_reads", "taxon_reads", "mm_tot", "mm_uniq", "rank", "t_id", "name"]
        keep_cols = ["clade_reads", "taxon_reads", "mm_tot", "mm_uniq", "t_id"]
    else: # LEGACY
        schema = {
            "column_1": pl.Float64, "column_2": pl.UInt32, "column_3": pl.UInt32,
            "column_4": pl.String, "column_5": pl.UInt32, "column_6": pl.String,
        }
        new_names = ["pct", "clade_reads", "taxon_reads", "rank", "t_id", "name"]
        keep_cols = ["clade_reads", "taxon_reads", "t_id"]

    df = pl.read_csv(
        file_path,
        has_header=False,
        separator="\t",
        schema=schema,
        new_columns=new_names
    )
    
    return df.select(keep_cols)

def gather_reports_logic(
    input_dir: Path,
    output_file: Path,
    recursive: bool = True,
    include_pattern: str = "*.report"
) -> None:
    """
    Finds and merges Kraken report files into a single Parquet file.
    Enforces format consistency (all 6-col or all 8-col).
    """
    search_path = "**/" + include_pattern if recursive else include_pattern
    report_files = list(input_dir.glob(search_path))
    
    if not report_files:
        raise FileNotFoundError(f"No files matching '{include_pattern}' found in {input_dir}")
        
    # 1. Determine Report Format and Batch Consistency
    report_format = detect_report_format(report_files[0])
    for f in report_files[1:]:
        fmt = detect_report_format(f)
        if fmt != report_format:
            raise ValueError(
                f"Mixed report formats detected. First file is {report_format}, "
                f"but '{f.name}' is {fmt}. Batch must be consistent."
            )

    # 2. Determine Data Standard (SWEBITS vs GENERIC)
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
    
    # 3. Process Files
    dfs = []
    for i, file_path in enumerate(report_files):
        sample_id = file_path.name.split('.')[0]
        df = parse_kraken_report(file_path, report_format)
        
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
    
    # 4. Finalize Schema and Sort
    sort_keys = ["year", "week", "sample_id", "t_id"] if is_swebits else ["sample_id", "t_id"]
    merged_df = merged_df.sort(sort_keys)
    
    # 5. Save with Metadata
    metadata = get_standard_metadata(
        file_type="REPORT_PARQUET", 
        source_path=input_dir,
        compression="zstd (level 3)",
        sorting=", ".join(sort_keys),
        data_standard=data_standard,
        report_format=report_format
    )
    
    write_parquet_with_metadata(
        merged_df, 
        output_file, 
        metadata, 
        compression="zstd", 
        compression_level=3
    )
