"""
sweetbits.metadata
Utilities for reading and writing Arrow-level metadata in Parquet files.
"""

import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, Any, Optional
from sweetbits import __version__

def get_standard_metadata(
    file_type: str, 
    source_path: Optional[Path] = None,
    compression: str = "None",
    sorting: str = "None",
    data_standard: str = "GENERIC",
    report_format: str = "HYPERLOGLOG"
) -> Dict[str, str]:
    """
    Generates the standard metadata dictionary for SweetBITS parquet files.

    Args:
        file_type           : String identifier for the file schema (e.g., 'REPORT_PARQUET').
        source_path         : Absolute path to the original source of the data.
        compression         : Description of the compression used.
        sorting             : Description of the column sorting applied.
        data_standard       : The detected standard profile ('SWEBITS' or 'GENERIC').
        report_format       : The detected input format ('HYPERLOGLOG' or 'LEGACY').

    Returns:
        A dictionary of metadata strings ready for Arrow schema injection.
    """
    args = sys.argv[1:]
    command_str = " ".join(args)
    
    # Identify the invocation command
    if not sys.argv[0].endswith("sweetbits"):
        invocation = f"python {sys.argv[0]} {command_str}".strip()
    else:
        invocation = f"sweetbits {command_str}".strip()

    metadata = {
        "sweetbits_version": __version__,
        "file_type": file_type,
        "execution_command": invocation,
        "creation_time": datetime.now().isoformat(),
        "compression": compression,
        "sorting": sorting,
        "source_path_abs": str(source_path.resolve()) if source_path else "Unknown",
        "data_standard": data_standard,
        "report_format": report_format
    }
    return metadata

def write_parquet_with_metadata(df: 'pl.DataFrame', output_path: Path, metadata: Dict[str, str], **kwargs):
    """
    Writes a Polars DataFrame to Parquet with custom file-level metadata.

    Args:
        df                  : The Polars DataFrame to save.
        output_path         : Target path for the Parquet file.
        metadata            : Dictionary of metadata to inject into the Arrow schema.
        **kwargs            : Additional arguments passed to pyarrow.parquet.write_table.
    """
    table = df.to_arrow()
    existing_meta = table.schema.metadata or {}
    merged_meta = {**existing_meta}
    for k, v in metadata.items():
        merged_meta[k.encode()] = str(v).encode()
        
    new_schema = table.schema.with_metadata(merged_meta)
    table = table.cast(new_schema)
    pq.write_table(table, output_path, **kwargs)

def read_parquet_metadata(file_path: Path) -> Dict[str, str]:
    """
    Reads the custom metadata from a Parquet file header.

    Args:
        file_path           : Path to the Parquet file.

    Returns:
        A dictionary of decoded metadata strings.
    """
    schema = pq.read_schema(file_path)
    if not schema.metadata:
        return {}
    return {k.decode(): v.decode() for k, v in schema.metadata.items()}
