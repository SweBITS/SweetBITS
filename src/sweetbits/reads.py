import polars as pl
import gzip
from pathlib import Path
from typing import Optional, List, Dict, Any
from joltax import JolTree
from sweetbits.utils import parse_sample_id

def format_short_name(scientific_name: str) -> str:
    """
    Formats a scientific name into a ShortName tag.
    - If >1 word: HomSap (first two words, 3 chars each, PascalCase)
    - If 1 word: Use whole word.
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
    """Checks if a (year, week) falls within the specified range."""
    current = (year, week)
    if year_start is not None:
        start = (year_start, week_start if week_start is not None else 0)
        if current < start: return False
    if year_end is not None:
        end = (year_end, week_end if week_end is not None else 99)
        if current > end: return False
    return True

def write_fastq_record(file_handle, read_id, seq, qual):
    """Writes a single FASTQ record to the handle."""
    file_handle.write(f"@{read_id}\n{seq}\n+\n{qual}\n".encode())

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
    week_end: Optional[int] = None
) -> Dict[str, Any]:
    """
    Streams KRAKEN_PARQUET files and extracts reads into FASTQ format.
    """
    # 1. Setup
    tree = JolTree.load(str(taxonomy_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discovery
    if input_path.is_dir():
        parquet_files = list(input_path.glob("*.parquet"))
    else:
        parquet_files = [input_path]
        
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at {input_path}")

    # Resolve Taxon Metadata
    taxon_info = {}
    for tid in tax_ids:
        name = tree.get_name(tid, strict=False) or f"Unknown{tid}"
        taxon_info[tid] = {
            "name": name,
            "short_name": format_short_name(name),
            "clade_members": set(tree.get_clade(tid)) if mode == "clade" else {tid}
        }

    # Combined File Handles (if active)
    combined_handles = {}
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
    
    try:
        for pfile in parquet_files:
            # Quick check of sample metadata before reading full file
            # Most SweBITS files have sample_id in the name
            sample_id = pfile.name.split('.')[0]
            try:
                info = parse_sample_id(sample_id)
                if not is_in_temporal_range(info['year'], info['week'], year_start, week_start, year_end, week_end):
                    continue
            except ValueError:
                # If filename parsing fails, we'll check inside the file
                pass

            # Stream the file
            lf = pl.scan_parquet(pfile)
            
            # Apply Temporal Filter inside Polars if possible
            if year_start:
                lf = lf.filter(pl.col("year") >= year_start)
                # Note: Exact (year, week) interval is harder in pure Polars filter 
                # without complex logic, but we'll filter more precisely in the loop.

            # Filter for requested taxa
            all_target_tids = set()
            for tid in tax_ids:
                all_target_tids.update(taxon_info[tid]["clade_members"])
            
            lf = lf.filter(pl.col("t_id").is_in(list(all_target_tids)))
            
            # Execute stream
            df = lf.collect(streaming=True)
            if df.is_empty():
                continue
                
            samples_processed += 1
            
            # Group by sample_id and t_id for file management
            for (sid, tid_internal), group in df.group_by(["sample_id", "t_id"]):
                # Precise temporal check for this specific sample record
                row = group.row(0, named=True)
                if not is_in_temporal_range(row['year'], row['week'], year_start, week_start, year_end, week_end):
                    continue
                
                # Determine which requested TaxID this hit belongs to
                for requested_tid in tax_ids:
                    if tid_internal in taxon_info[requested_tid]["clade_members"]:
                        tmeta = taxon_info[requested_tid]
                        
                        # Get or create handles
                        if combine_samples:
                            handle_key = requested_tid
                            if handle_key not in combined_handles:
                                fname_base = f"combined_{mode}_{requested_tid}_{tmeta['short_name']}{range_tag}"
                                combined_handles[handle_key] = (
                                    gzip.open(output_dir / f"{fname_base}_R1.fastq.gz", "ab"),
                                    gzip.open(output_dir / f"{fname_base}_R2.fastq.gz", "ab")
                                )
                            h1, h2 = combined_handles[handle_key]
                        else:
                            fname_base = f"{sid}_{mode}_{requested_tid}_{tmeta['short_name']}"
                            h1 = gzip.open(output_dir / f"{fname_base}_R1.fastq.gz", "ab")
                            h2 = gzip.open(output_dir / f"{fname_base}_R2.fastq.gz", "ab")

                        # Write reads
                        for read in group.iter_rows(named=True):
                            write_fastq_record(h1, read['read_id'], read['r1_seq'], read['r1_qual'])
                            write_fastq_record(h2, read['read_id'], read['r2_seq'], read['r2_qual'])
                            total_reads_extracted += 1
                        
                        if not combine_samples:
                            h1.close()
                            h2.close()

    finally:
        for h1, h2 in combined_handles.values():
            h1.close()
            h2.close()

    return {
        "samples_processed": samples_processed,
        "reads_extracted": total_reads_extracted,
        "output_dir": str(output_dir)
    }
