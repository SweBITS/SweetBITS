"""
sweetbits.convert
Logic for ingestion of Kraken and FASTQ files into KRAKEN_PARQUET format.
"""

import subprocess
import os
import tempfile
import click
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any, Optional, Iterator, Tuple

from sweetbits.utils import parse_sample_id, get_sample_info
from sweetbits.metadata import get_standard_metadata

def _open_text_stream(path: Path):
    """
    Opens a text stream, using an OS-level gzip subprocess if necessary for maximum performance.

    Args:
        path : Path to the file to open (supports .gz).

    Returns:
        A tuple of (stream, process). If the file is not gzipped, process will be None.
    """
    if path.suffix == ".gz":
        # Bypass Python GIL and use OS-level decompression
        proc = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE, text=True, bufsize=1024*1024)
        return proc.stdout, proc
    else:
        f = open(path, "rt", buffering=1024*1024)
        return f, None

def _fastq_iterator(f_stream) -> Iterator[Tuple[str, str, str]]:
    """
    Yields parsed records from a FASTQ stream.

    Args:
        f_stream : An active text iterator reading a FASTQ file.

    Yields:
        A tuple containing (read_id, sequence, quality_string). 
        The read_id is stripped of its leading '@' and any pair suffixes 
        (e.g., '/1' or '/2') to match the Kraken output format.
    """
    try:
        while True:
            header = next(f_stream)
            seq = next(f_stream).rstrip('\n')
            next(f_stream)  # skip '+'
            qual = next(f_stream).rstrip('\n')
            
            # Extract read_id: drop '@', take first word
            read_id = header.split()[0][1:]
            
            # Handle standard Illumina pair suffixes
            if read_id.endswith('/1') or read_id.endswith('/2'):
                read_id = read_id[:-2]
                
            yield read_id, seq, qual
    except StopIteration:
        pass

def convert_kraken_logic(
    kraken_file: Path,
    output_file: Path,
    r1_file: Optional[Path] = None,
    r2_file: Optional[Path] = None,
    cores: Optional[int] = None
) -> Dict[str, Any]:
    """
    Converts Kraken output and FASTQ files into a highly compressed, sorted KRAKEN_PARQUET.
    
    Uses a memory-safe two-pointer streaming algorithm for the Left Join, followed
    by an out-of-core Rust/Polars sort phase.

    Args:
        kraken_file : Path to the Kraken read-by-read output (.txt or .gz).
        output_file : Path where the final Parquet will be saved.
        r1_file     : Optional path to the R1 FASTQ file (.fastq or .gz).
        r2_file     : Optional path to the R2 FASTQ file (.fastq or .gz).
        cores       : Number of threads to assign to Polars for out-of-core sorting.

    Returns:
        A dictionary containing processing statistics:
        - 'records_processed': Number of reads parsed.
        - 'has_fastq'        : Boolean flag if sequence data is included.
        - 'data_standard'    : The standard (SWEBITS/GENERIC) applied to the file.
        - 'output_file'      : Path to the generated Parquet file.

    Raises:
        RuntimeError : If the FASTQ files are fundamentally out of sync with the Kraken file.
    """
    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)
        
    # 1. Determine Data Standard
    # Check if the filename matches SweBITS patterns to inject temporal columns
    info = get_sample_info(kraken_file.name)
    data_standard = info["data_standard"]
    sample_id = info["sample_id"]
    year, week = info["year"], info["week"]

    # Enforce project requirement for paired-end data
    if (r1_file is not None) != (r2_file is not None):
        raise ValueError(
            "SweetBITS requires paired-end data. Both --r1 and --r2 FASTQ files "
            "must be provided, or neither to create a skinny Parquet."
        )
    has_fastq = (r1_file is not None)
    
    # Log the conversion mode
    if has_fastq:
        click.secho("Mode: FAT Parquet (FASTQ sequences and quality scores included).", fg="cyan", err=True)
        click.secho("Info: You can extract reads directly from this Parquet file using 'extract-reads'.", fg="cyan", err=True)
    else:
        click.secho("Mode: SKINNY Parquet (Taxonomic info only, sequences omitted).", fg="cyan", err=True)
        click.secho("Info: The 'extract-reads' command will only output read ID lists for this file.", fg="cyan", err=True)
    
    # 2. Stream Initialization
    # We use independent OS-level decompression streams to avoid the Python GIL
    k_stream, k_proc = _open_text_stream(kraken_file)
    r1_stream, r1_proc, r2_stream, r2_proc = None, None, None, None
    r1_iter, r2_iter = None, None
    
    if has_fastq:
        r1_stream, r1_proc = _open_text_stream(r1_file)
        r1_iter = _fastq_iterator(r1_stream)
        r2_stream, r2_proc = _open_text_stream(r2_file)
        r2_iter = _fastq_iterator(r2_stream)

    curr_r1 = next(r1_iter) if r1_iter else None
    curr_r2 = next(r2_iter) if r2_iter else None

    CHUNK_SIZE = 500_000
    records_processed = 0
    
    # 3. Schema Definition
    # We strictly enforce datatypes at ingestion to minimize disk footprint
    schema_fields = [
        ("sample_id", pa.string()),
        ("read_id", pa.string()),
        ("t_id", pa.uint32()),
        ("mhg", pa.uint8()),
        ("r1_len", pa.uint8()),
        ("r2_len", pa.uint8()),
        ("total_len", pa.uint16()),
        ("kmer_string", pa.string())
    ]
    if data_standard == "SWEBITS":
        schema_fields.append(("year", pa.uint16()))
        schema_fields.append(("week", pa.uint8()))
        
    if has_fastq:
        schema_fields.extend([
            ("r1_seq", pa.string()),
            ("r1_qual", pa.string()),
            ("r2_seq", pa.string()),
            ("r2_qual", pa.string())
        ])
        
    schema = pa.schema(schema_fields)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_unsorted = Path(tmpdir) / "unsorted.parquet"
        writer = pq.ParquetWriter(tmp_unsorted, schema, compression='NONE')
        
        # 4. Two-Pointer Streaming Logic
        # The Kraken file acts as the absolute source of truth. If a FASTQ read is missing 
        # (e.g., host depletion), we insert nulls but keep the taxonomic classification.
        try:
            while True:
                chunk_data = {f[0]: [] for f in schema_fields}
                lines_read = 0
                
                while lines_read < CHUNK_SIZE:
                    line = k_stream.readline()
                    if not line:
                        break
                        
                    lines_read += 1
                    parts = line.rstrip('\n').split('\t')
                    
                    # Parse SweBITS/Kraken read-by-read format
                    read_id = parts[1]
                    t_id = int(parts[2])
                    
                    lens = parts[3].split('|')
                    r1_len = int(lens[0])
                    r2_len = int(lens[1]) if len(lens) > 1 else 0
                    total_len = r1_len + r2_len
                    
                    try:
                        mhg = int(parts[4])
                    except (IndexError, ValueError):
                        mhg = 0
                        
                    kmer_string = parts[5] if len(parts) > 5 else ""
                    
                    chunk_data["sample_id"].append(sample_id)
                    chunk_data["read_id"].append(read_id)
                    chunk_data["t_id"].append(t_id)
                    chunk_data["mhg"].append(mhg)
                    chunk_data["r1_len"].append(r1_len)
                    chunk_data["r2_len"].append(r2_len)
                    chunk_data["total_len"].append(total_len)
                    chunk_data["kmer_string"].append(kmer_string)
                    
                    if data_standard == "SWEBITS":
                        chunk_data["year"].append(year)
                        chunk_data["week"].append(week)
                        
                    if has_fastq:
                        r1_s, r1_q = None, None
                        r2_s, r2_q = None, None
                        
                        # Left Join: Match FASTQ sequences if IDs align perfectly
                        if curr_r1 and curr_r1[0] == read_id:
                            _, r1_s, r1_q = curr_r1
                            try:
                                curr_r1 = next(r1_iter)
                            except StopIteration:
                                curr_r1 = None
                                
                        if curr_r2 and curr_r2[0] == read_id:
                            _, r2_s, r2_q = curr_r2
                            try:
                                curr_r2 = next(r2_iter)
                            except StopIteration:
                                curr_r2 = None
                                
                        chunk_data["r1_seq"].append(r1_s)
                        chunk_data["r1_qual"].append(r1_q)
                        chunk_data["r2_seq"].append(r2_s)
                        chunk_data["r2_qual"].append(r2_q)
                        
                    records_processed += 1
                
                if lines_read == 0:
                    break
                    
                table = pa.Table.from_pydict(chunk_data, schema=schema)
                writer.write_table(table)
                
        finally:
            writer.close()
            k_stream.close()
            if k_proc: k_proc.wait()
            if r1_stream: r1_stream.close()
            if r1_proc: r1_proc.wait()
            if r2_stream: r2_stream.close()
            if r2_proc: r2_proc.wait()

        # 5. Synchronicity Audit
        # If the Kraken file is exhausted but FASTQ files still have reads, 
        # the pipelines drifted out of sync upstream.
        if has_fastq and (curr_r1 is not None or curr_r2 is not None):
            raise RuntimeError(
                "FASTQ files are out of sync with the Kraken report or contain reads "
                "not present in the Kraken output. Ensure downstream tools preserved read order."
            )

        # 6. Phase 2: Sort (Out-of-Core)
        # Sort by TaxID to maximize Run-Length Encoding (RLE) during Parquet zstd compression
        tmp_sorted = Path(tmpdir) / "sorted.parquet"
        lf = pl.scan_parquet(tmp_unsorted).sort("t_id")
        lf.sink_parquet(tmp_sorted, compression="uncompressed")
        
        # 7. Phase 3: Metadata Injection & Final Compression
        meta = get_standard_metadata(
            file_type="KRAKEN_PARQUET",
            source_path=kraken_file,
            compression="zstd",
            sorting="t_id",
            data_standard=data_standard,
            report_format="UNKNOWN"
        )
        meta["has_fastq"] = "True" if has_fastq else "False"
        
        sorted_pf = pq.ParquetFile(tmp_sorted)
        
        existing_meta = sorted_pf.schema_arrow.metadata or {}
        merged_meta = {**existing_meta}
        for k, v in meta.items():
            merged_meta[k.encode()] = str(v).encode()
            
        new_schema = sorted_pf.schema_arrow.with_metadata(merged_meta)
        
        with pq.ParquetWriter(output_file, new_schema, compression="zstd", compression_level=3) as final_writer:
            for batch in sorted_pf.iter_batches(batch_size=100_000):
                final_writer.write_batch(batch)

    return {
        "records_processed": records_processed,
        "has_fastq": has_fastq,
        "data_standard": data_standard,
        "output_file": str(output_file)
    }
