"""
sweetbits.reads
Logic for streaming and extracting reads from Kraken-annotated Parquet files.
"""

import polars as pl
import gzip
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, IO, Tuple
from collections import OrderedDict
from joltax import JolTree
from sweetbits.utils import parse_sample_id, get_sample_info
from sweetbits.metadata import validate_sweetbits_file

def format_short_name(scientific_name: str) -> str:
    """
    Formats a scientific name into a ShortName tag.

    Args:
        scientific_name : The full scientific name string.

    Returns:
        A condensed string (e.g., 'HomSap' for 'Homo sapiens').
    """
    words = scientific_name.split()
    if len(words) > 1:
        # Take first two words, first 3 chars each
        w1 = words[0][:3].capitalize()
        w2 = words[1][:3].capitalize()
        return f"{w1}{w2}"
    return scientific_name

def is_in_temporal_range(
    year: int, 
    week: int, 
    year_start: Optional[int] = None, 
    week_start: Optional[int] = None,
    year_end: Optional[int] = None, 
    week_end: Optional[int] = None
) -> bool:
    """
    Checks if a (year, week) falls within the specified range.

    Args:
        year       : The ISO year to check.
        week       : The ISO week to check.
        year_start : Optional start year filter.
        week_start : Optional start week filter.
        year_end   : Optional end year filter.
        week_end   : Optional end week filter.

    Returns:
        True if the point is within the inclusive interval.
    """
    current = (year, week)
    if year_start is not None:
        start = (year_start, week_start if week_start is not None else 0)
        if current < start: return False
    if year_end is not None:
        end = (year_end, week_end if week_end is not None else 99)
        if current > end: return False
    return True

class FastqHandleManager:
    """
    Manages open GZIP or Text handles using an LRU cache strategy.
    
    This prevents 'Too many open files' OS errors while minimizing the 
    overhead of opening/closing streams for common targets.
    """
    def __init__(self, output_dir: Path, max_handles: int = 400, extension: str = ".fastq.gz"):
        self.output_dir = output_dir
        self.max_handles = max_handles
        self.extension = extension
        self.handles: OrderedDict[str, IO] = OrderedDict()

    def get_handle(self, name: str) -> IO:
        """Retrieves an open handle, opening it if necessary and enforcing limits."""
        if name in self.handles:
            self.handles.move_to_end(name)
            return self.handles[name]
        
        # Enforce OS file limit by closing the Least Recently Used handle
        if len(self.handles) >= self.max_handles:
            _, oldest_handle = self.handles.popitem(last=False)
            oldest_handle.close()
            
        # If extension is fastq.gz, use gzip, otherwise use standard open
        if self.extension.endswith(".gz"):
            handle = gzip.open(self.output_dir / f"{name}{self.extension}", "ab")
        else:
            handle = open(self.output_dir / f"{name}{self.extension}", "ab")
            
        self.handles[name] = handle
        return handle

    def close_all(self):
        """Closes all currently open handles."""
        for h in self.handles.values():
            h.close()
        self.handles.clear()

def extract_reads_logic(
    input_path: Path,
    taxonomy_dir: Path,
    tax_ids: List[int],
    output_dir: Path,
    mode: str = "clade",
    combine_samples: bool = False,
    year_start: Optional[int] = None,
    week_start: Optional[int] = None,
    year_end: Optional[int] = None,
    week_end: Optional[int] = None,
    cores: Optional[int] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Streams KRAKEN_PARQUET files and extracts reads into FASTQ or text format.

    Args:
        input_path      : Path to a single Parquet file or a directory of files.
        taxonomy_dir    : Path to the JolTax cache directory.
        tax_ids         : List of TaxIDs to extract reads for.
        output_dir      : Directory where output files will be saved.
        mode            : Extraction mode ('taxon' or 'clade').
        combine_samples : Whether to merge reads from all samples into one file per TaxID.
        year_start      : Optional start year for filtering.
        week_start      : Optional start week for filtering.
        year_end        : Optional end year for filtering.
        week_end        : Optional end week for filtering.
        cores           : Number of CPU cores to use for Polars operations.
        overwrite       : Whether to overwrite existing files in output_dir.

    Returns:
        A dictionary containing extraction statistics.

    Raises:
        FileExistsError : If output directory already exists and overwrite is False.
    """
    # 1. Setup
    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)

    # Overwrite Protection
    if not overwrite and output_dir.exists():
        # Check if the directory is empty or if it's a custom folder
        is_current_dir = output_dir.resolve() == Path.cwd().resolve()
        if not is_current_dir or any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory '{output_dir}' already exists and is not empty. "
                "Use --overwrite to replace existing data."
            )

    tree = JolTree.load(str(taxonomy_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_dir():
        parquet_files = sorted(list(input_path.glob("*.parquet")))
    else:
        parquet_files = [input_path]
        
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at {input_path}")

    # Map requested TaxIDs to their metadata and clade members
    taxon_meta = {}
    all_target_tids = set()
    for tid in tax_ids:
        name = tree.get_name(tid, strict=False) or f"Unknown{tid}"
        members = set(tree.get_clade(tid)) if mode == "clade" else {tid}
        taxon_meta[tid] = {
            "short_name": format_short_name(name),
            "members": members
        }
        all_target_tids.update(members)

    range_tag = ""
    if combine_samples and (year_start or year_end):
        ys = year_start or "Start"
        ws = f"W{week_start:02}" if week_start else ""
        ye = year_end or "End"
        we = f"W{week_end:02}" if week_end else ""
        range_tag = f"_{ys}{ws}-to-{ye}{we}"

    # 2. Processing
    total_reads_extracted = 0
    samples_processed = 0
    handle_manager = None
    
    try:
        for pfile in parquet_files:
            # FAST-FAIL: Quick filename-based temporal check before scanning Parquet
            info = get_sample_info(pfile.name)
            if info["data_standard"] == "SWEBITS":
                if not is_in_temporal_range(info['year'], info['week'], year_start, week_start, year_end, week_end):
                    continue

            # Validate metadata and columns via JSON companion
            schema = pl.scan_parquet(pfile).collect_schema()
            metadata = validate_sweetbits_file(pfile, expected_type="KRAKEN_PARQUET")
            has_fastq = metadata.get("has_fastq") == "True"
            
            required_cols = ["sample_id", "read_id", "t_id", "year", "week"]
            if has_fastq:
                required_cols.extend(["r1_seq", "r1_qual", "r2_seq", "r2_qual"])
                ext = ".fastq.gz"
            else:
                ext = ".txt"

            # Check if required columns are in schema
            missing = [c for c in required_cols if c not in schema]
            if missing:
                raise ValueError(f"File {pfile.name} is missing required columns: {missing}")

            # Initialize handle manager if needed
            if handle_manager is None or handle_manager.extension != ext:
                if handle_manager: handle_manager.close_all()
                handle_manager = FastqHandleManager(output_dir, extension=ext)
                
                # Pre-initialize combined handles
                if combine_samples:
                    for tid in tax_ids:
                        tmeta = taxon_meta[tid]
                        fname_base = f"combined_{mode}_{tid}_{tmeta['short_name']}{range_tag}"
                        if has_fastq:
                            handle_manager.get_handle(f"{fname_base}_R1")
                            handle_manager.get_handle(f"{fname_base}_R2")
                        else:
                            handle_manager.get_handle(fname_base)

            # Pre-initialize per-sample handles
            if not combine_samples:
                for tid in tax_ids:
                    tmeta = taxon_meta[tid]
                    fname_base = f"{info['sample_id']}_{mode}_{tid}_{tmeta['short_name']}"
                    if has_fastq:
                        handle_manager.get_handle(f"{fname_base}_R1")
                        handle_manager.get_handle(f"{fname_base}_R2")
                    else:
                        handle_manager.get_handle(fname_base)

            # 3. Stream Matching Records
            lf = pl.scan_parquet(pfile)
            lf = lf.filter(pl.col("t_id").is_in(list(all_target_tids)))
            
            # Apply temporal filters at scan level for SWEBITS standard
            data_standard = metadata.get("data_standard", "GENERIC")
            if data_standard == "SWEBITS":
                if year_start: lf = lf.filter(pl.col("year") >= year_start)
                if year_end:   lf = lf.filter(pl.col("year") <= year_end)
            
            df = lf.collect(engine="streaming")
            if df.is_empty():
                continue
                
            samples_processed += 1
            
            # Group by Sample and TaxID
            for (sid, tid_internal), group in df.group_by(["sample_id", "t_id"]):
                # Precise temporal check for the actual data
                if data_standard == "SWEBITS":
                    row_meta = group.row(0, named=True)
                    if not is_in_temporal_range(row_meta['year'], row_meta['week'], year_start, week_start, year_end, week_end):
                        continue
                
                # Identify every requested clade that this TaxID belongs to
                matching_requests = [
                    req_tid for req_tid in tax_ids 
                    if tid_internal in taxon_meta[req_tid]["members"]
                ]
                
                if not matching_requests:
                    continue

                # MEMORY SAFEGUARD: Process massive groups in chunks of 50,000 reads.
                CHUNK_SIZE = 50_000
                for chunk in group.iter_slices(CHUNK_SIZE):
                    if has_fastq:
                        # Filter out depleted reads (those without physical sequence data)
                        chunk = chunk.filter(pl.col("r1_seq").is_not_null())
                        if chunk.is_empty():
                            continue

                    num_reads_in_chunk = chunk.height
                    
                    if has_fastq:
                        # PERFORMANCE: Pre-compile FASTQ strings into binary byte-blocks.
                        records = list(zip(
                            chunk["read_id"].to_list(),
                            chunk["r1_seq"].to_list(),
                            chunk["r1_qual"].to_list(),
                            chunk["r2_seq"].to_list(),
                            chunk["r2_qual"].to_list()
                        ))
                        
                        block_r1 = "".join([f"@{rid}\n{r1s}\n+\n{r1q}\n" for rid, r1s, r1q, r2s, r2q in records]).encode()
                        block_r2 = "".join([f"@{rid}\n{r2s}\n+\n{r2q}\n" for rid, r1s, r1q, r2s, r2q in records]).encode()
                    else:
                        # SKINNY MODE: Just pre-compile ID list
                        block_ids = "\n".join(chunk["read_id"].to_list()).encode() + b"\n"

                    for requested_tid in matching_requests:
                        tmeta = taxon_meta[requested_tid]
                        
                        if combine_samples:
                            fname_base = f"combined_{mode}_{requested_tid}_{tmeta['short_name']}{range_tag}"
                        else:
                            fname_base = f"{sid}_{mode}_{requested_tid}_{tmeta['short_name']}"
                        
                        if has_fastq:
                            h1 = handle_manager.get_handle(f"{fname_base}_R1")
                            h2 = handle_manager.get_handle(f"{fname_base}_R2")
                            h1.write(block_r1)
                            h2.write(block_r2)
                        else:
                            h = handle_manager.get_handle(fname_base)
                            h.write(block_ids)
                            
                        total_reads_extracted += num_reads_in_chunk
            
            # If not combining, close handles after each sample to keep resources tight
            if not combine_samples:
                handle_manager.close_all()

    finally:
        if handle_manager:
            handle_manager.close_all()

    return {
        "samples_processed": samples_processed,
        "records_extracted": total_reads_extracted,
        "output_dir": str(output_dir)
    }
