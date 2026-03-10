"""
sweetbits.convert
Logic for ingestion of Kraken and FASTQ files into KRAKEN_PARQUET format.
"""

import subprocess
import os
import tempfile
import click
import polars as pl
import gc
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Iterator, Tuple

from sweetbits.utils import parse_sample_id, get_sample_info, check_write_permission
from sweetbits.metadata import get_standard_metadata, save_companion_metadata

# Tweak Polars for large-scale data handling
# Setting a smaller chunk size for streaming operations to reduce peak memory pressure
pl.Config.set_streaming_chunk_size(50_000)

def _log_mem(label: str):
    """Logs current Resident Set Size (RSS) in MB."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    click.secho(f"  [RAM] {label:20}: {mem_mb:,.2f} MB", fg="bright_black", err=True)

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
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Converts Kraken output and FASTQ files into a highly compressed, sorted KRAKEN_PARQUET.
    
    Uses a memory-safe two-pointer streaming algorithm for the Left Join, followed
    by an out-of-core Polars sort phase.

    Args:
        kraken_file : Path to the Kraken read-by-read output (.txt or .gz).
        output_file : Path where the final Parquet will be saved.
        r1_file     : Optional path to the R1 FASTQ file (.fastq or .gz).
        r2_file     : Optional path to the R2 FASTQ file (.fastq or .gz).
        cores       : Number of threads to assign to Polars for out-of-core sorting.
        overwrite   : Whether to overwrite the output file if it exists.

    Returns:
        A dictionary containing processing statistics:
        - 'records_processed': Number of reads parsed.
        - 'has_fastq'        : Boolean flag if sequence data is included.
        - 'data_standard'    : The standard (SWEBITS/GENERIC) applied to the file.
        - 'output_file'      : Path to the generated Parquet file.

    Raises:
        RuntimeError    : If the FASTQ files are fundamentally out of sync with the Kraken file.
        FileExistsError : If output_file exists and overwrite is False.
    """
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file '{output_file}' already exists. Use --overwrite to replace it.")

    # 0. Early Permission Check
    check_write_permission(output_file)

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
        click.secho(click.style("Info: You can extract reads directly from this Parquet file using ", fg="cyan") + click.style("'produce reads'", fg="cyan", bold=True) + click.style(".", fg="cyan"), err=True)
    else:
        click.secho("Mode: SKINNY Parquet (Taxonomic info only, sequences omitted).", fg="cyan", err=True)
        click.secho(click.style("Info: The ", fg="cyan") + click.style("'produce reads'", fg="cyan", bold=True) + click.style(" command will only output read ID lists for this file.", fg="cyan"), err=True)
    
    _log_mem("Start of Conversion")

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

    curr_r1 = None
    curr_r2 = None
    if has_fastq:
        try:
            curr_r1 = next(r1_iter)
            curr_r2 = next(r2_iter)
        except StopIteration:
            pass

    CHUNK_SIZE = 200_000
    records_processed = 0
    matched_fastq_count = 0
    last_matched_id = "None"
    
    # 3. Schema Definition
    # We define Polars schema dict to explicitly construct DataFrames
    schema_fields = {
        "sample_id": pl.Categorical,
        "read_id": pl.String,
        "t_id": pl.UInt32,
        "mhg": pl.UInt8,
        "r1_len": pl.UInt8,
        "r2_len": pl.UInt8,
        "total_len": pl.UInt16,
        "kmer_string": pl.String
    }
    if data_standard == "SWEBITS":
        schema_fields["year"] = pl.UInt16
        schema_fields["week"] = pl.UInt8
        
    if has_fastq:
        schema_fields["r1_seq"] = pl.String
        schema_fields["r1_qual"] = pl.String
        schema_fields["r2_seq"] = pl.String
        schema_fields["r2_qual"] = pl.String

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_chunks_dir = Path(tmpdir) / "chunks"
        tmp_chunks_dir.mkdir()
        
        # 4. Two-Pointer Streaming Logic
        # The Kraken file acts as the absolute source of truth. If a FASTQ read is missing 
        # (e.g., host depletion), we insert nulls but keep the taxonomic classification.
        click.secho("Phase 1/2: Ingesting data and synchronizing streams...", fg="cyan", err=True)
        try:
            chunk_idx = 0
            # Use StringCache across ingestion and writing chunks so Categorical dicts are unified
            with pl.StringCache():
                while True:
                    chunk_data = {col: [] for col in schema_fields}
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
                                
                                matched_fastq_count += 1
                                last_matched_id = read_id
                                    
                            chunk_data["r1_seq"].append(r1_s)
                            chunk_data["r1_qual"].append(r1_q)
                            chunk_data["r2_seq"].append(r2_s)
                            chunk_data["r2_qual"].append(r2_q)
                            
                        records_processed += 1
                    
                    if lines_read == 0:
                        break
                        
                    # Create DataFrame and apply specific typing
                    df = pl.DataFrame(chunk_data, schema_overrides=schema_fields, strict=False)
                    
                    # Write snappy-compressed parquet chunk to tempdir
                    chunk_path = tmp_chunks_dir / f"chunk_{chunk_idx:05d}.parquet"
                    df.write_parquet(chunk_path, compression="snappy")
                    chunk_idx += 1
                    
                    # Provide periodic feedback
                    if records_processed % 1_000_000 == 0:
                        _log_mem(f"Ingested {records_processed//1_000_000}M reads")
                
        finally:
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
            msg = (
                "FASTQ files are out of sync with the Kraken report or contain extra reads. "
                f"Total Kraken reads processed: {records_processed}. "
                f"FASTQ reads successfully matched: {matched_fastq_count}. "
                f"Last matched Read ID: '{last_matched_id}'. "
            )
            if curr_r1:
                msg += f"First unmatched FASTQ ID: '{curr_r1[0]}'. "
            
            msg += "Ensure downstream tools (depletion/cleaning) preserved read order and did not add new reads."
            raise RuntimeError(msg)

        _log_mem("End of Phase 1")
        gc.collect()
        _log_mem("After GC")

        # 6. Phase 2: Sort and Sinking (Out-of-Core)
        # Sort by TaxID to maximize Run-Length Encoding (RLE) during Parquet zstd compression.
        # We temporarily throttle cores during sort to cap memory-hungry per-thread buffers.
        sort_cores = min(cores if cores else os.cpu_count(), 4)
        orig_threads = os.environ.get("POLARS_MAX_THREADS")
        os.environ["POLARS_MAX_THREADS"] = str(sort_cores)
        
        click.secho(f"Phase 2/2: Polars out-of-core sorting and zstd compression (throttled to {sort_cores} cores)...", fg="cyan", err=True)
        
        with pl.StringCache():
            lf = pl.scan_parquet(tmp_chunks_dir / "*.parquet").sort("t_id")
            lf.sink_parquet(output_file, compression="zstd", compression_level=3)
        
        # Restore original thread count
        if orig_threads:
            os.environ["POLARS_MAX_THREADS"] = orig_threads
        else:
            del os.environ["POLARS_MAX_THREADS"]

        _log_mem("End of Phase 2 (Output written)")
        
        # Write the companion JSON metadata file
        meta = get_standard_metadata(
            file_type="KRAKEN_PARQUET",
            source_path=kraken_file,
            compression="zstd (level 3)",
            sorting="t_id",
            data_standard=data_standard,
            report_format="UNKNOWN"
        )
        meta["has_fastq"] = "True" if has_fastq else "False"
        save_companion_metadata(output_file, meta)

    _log_mem("End of Conversion")
    
    click.secho("Done!", fg="cyan", bold=True, err=True)
    
    return {
        "records_processed": records_processed,
        "has_fastq": has_fastq,
        "data_standard": data_standard,
        "output_file": str(output_file)
    }
