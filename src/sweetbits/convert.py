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

from sweetbits.utils import parse_sample_id
from sweetbits.metadata import get_standard_metadata

def _open_text_stream(path: Path):
    """Opens a text stream, using gzip subprocess if necessary for performance."""
    if path.suffix == ".gz":
        # Bypass Python GIL and use OS-level decompression
        proc = subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE, text=True, bufsize=1024*1024)
        return proc.stdout, proc
    else:
        f = open(path, "rt", buffering=1024*1024)
        return f, None

def _fastq_iterator(f_stream) -> Iterator[Tuple[str, str, str]]:
    """Yields (read_id, seq, qual) from a FASTQ stream."""
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
    no_fastq: bool = False,
    cores: Optional[int] = None
) -> Dict[str, Any]:
    """
    Converts Kraken output and FASTQ files into a highly compressed, sorted KRAKEN_PARQUET.
    
    Uses a memory-safe two-pointer streaming algorithm for the Left Join, followed
    by an out-of-core Rust/Polars sort phase.
    """
    if cores:
        os.environ["POLARS_MAX_THREADS"] = str(cores)
        
    # Determine sample ID and standard
    sample_id_guess = kraken_file.name.split('.')[0]
    try:
        info = parse_sample_id(sample_id_guess)
        data_standard = "SWEBITS"
        year, week = info["year"], info["week"]
        sample_id = sample_id_guess
    except ValueError:
        data_standard = "GENERIC"
        year, week = 0, 0
        sample_id = sample_id_guess

    has_fastq = False if no_fastq else (r1_file is not None)
    
    # Setup Streams
    k_stream, k_proc = _open_text_stream(kraken_file)
    r1_stream, r1_proc, r2_stream, r2_proc = None, None, None, None
    r1_iter, r2_iter = None, None
    
    if has_fastq:
        r1_stream, r1_proc = _open_text_stream(r1_file)
        r1_iter = _fastq_iterator(r1_stream)
        if r2_file:
            r2_stream, r2_proc = _open_text_stream(r2_file)
            r2_iter = _fastq_iterator(r2_stream)

    curr_r1 = next(r1_iter) if r1_iter else None
    curr_r2 = next(r2_iter) if r2_iter else None

    CHUNK_SIZE = 500_000
    records_processed = 0
    
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
                    
                    # Kraken 6-col (with MHG) or 8-col format parsing
                    # Our schema expects: Class, read_id, t_id, len(s), mhg, kmer
                    # Wait, standard Kraken is: Class, ID, t_id, len, LCA_kmer
                    # If SweBITS format has MHG as col 5 and kmer as col 6
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

        # Synchronicity Audit
        if has_fastq and (curr_r1 is not None or curr_r2 is not None):
            raise RuntimeError(
                "FASTQ files are out of sync with the Kraken report or contain reads "
                "not present in the Kraken output. Ensure downstream tools preserved read order."
            )

        # Phase 2: Sort (Out-of-Core)
        tmp_sorted = Path(tmpdir) / "sorted.parquet"
        lf = pl.scan_parquet(tmp_unsorted).sort("t_id")
        # Sink it uncompressed to save CPU time before the final PyArrow pass
        lf.sink_parquet(tmp_sorted, compression="uncompressed")
        
        # Phase 3: Metadata Injection & Compression
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
        
        # Inject metadata into Arrow schema
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
