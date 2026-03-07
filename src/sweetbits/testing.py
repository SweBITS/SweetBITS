"""
sweetbits.testing
Utilities for generating mock data for testing SweetBITS tools.
"""

import polars as pl
import numpy as np
import random
from typing import Optional, List
from pathlib import Path
from sweetbits.utils import parse_sample_id
from sweetbits.metadata import get_standard_metadata, write_parquet_with_metadata

# ... (Existing generate_mock_kraken_parquet and others) ...

def generate_mock_kraken_report_file(output_path: Path, format: str = "HYPERLOGLOG"):
    """
    Generates a mock Kraken report file.
    
    Args:
        output_path: Where to save the file.
        format: 'HYPERLOGLOG' (8 columns) or 'LEGACY' (6 columns).
    """
    if format == "HYPERLOGLOG":
        # Percentage, Clade reads, Taxon reads, MM_total, MM_uniq, Rank, TaxID, Name
        lines = [
            "100.0\t1000\t0\t5000\t2000\tR\t1\troot",
            "80.0\t800\t100\t4000\t1500\tD\t2\tBacteria",
            "20.0\t200\t50\t1000\t500\tD\t2759\tEukaryota",
        ]
    else: # LEGACY
        # Percentage, Clade reads, Taxon reads, Rank, TaxID, Name
        lines = [
            "100.0\t1000\t0\tR\t1\troot",
            "80.0\t800\t100\tD\t2\tBacteria",
            "20.0\t200\t50\tD\t2759\tEukaryota",
        ]
        
    with open(output_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

# Copying back the rest of the file to ensure it's complete
def generate_random_dna(length: int) -> str:
    return "".join(random.choice("ACGT") for _ in range(length))

def generate_random_qual(length: int) -> str:
    return "".join(chr(random.randint(33, 75)) for _ in range(length))

def generate_mock_kraken_parquet(
    sample_id: str, 
    num_reads: int = 100, 
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """Generates mock <KRAKEN_PARQUET> data."""
    sample_info = parse_sample_id(sample_id)
    data = []
    for i in range(num_reads):
        r1_len = random.randint(50, 150)
        r2_len = random.randint(50, 150)
        total_len = r1_len + r2_len
        k = 35
        r1_kmers = max(0, r1_len - k + 1)
        r2_kmers = max(0, r2_len - k + 1)
        kmers_total = max(1, r1_kmers + r2_kmers)
        t_id = random.choice([9606, 10090, 5000001, 5000002])
        data.append({
            "sample_id": sample_id, "year": sample_info["year"], "week": sample_info["week"],
            "read_id": f"read_{i}", "r1_qual": generate_random_qual(r1_len), "r2_qual": generate_random_qual(r2_len),
            "r1_seq": generate_random_dna(r1_len), "r2_seq": generate_random_dna(r2_len),
            "r1_len": r1_len, "r2_len": r2_len, "total_len": total_len, "t_id": t_id,
            "mhg": random.randint(1, 10), "kmer_string": f"{t_id}:10 0:5",
            "kmers_total": kmers_total, "kmers_ambig": 0, "kmers_clade": 10, "kmers_lineage": 10, "kmers_misclassified": 0,
            "clade_ratio": 0.5, "lineage_ratio": 0.5, "misclassified_ratio": 0.0,
        })
    df = pl.DataFrame(data).with_columns([
        pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8),
        pl.col("r1_len").cast(pl.UInt8), pl.col("r2_len").cast(pl.UInt8),
        pl.col("total_len").cast(pl.UInt16), pl.col("t_id").cast(pl.UInt32),
        pl.col("mhg").cast(pl.UInt8), pl.col("kmers_total").cast(pl.UInt8),
        pl.col("kmers_ambig").cast(pl.UInt8), pl.col("kmers_clade").cast(pl.UInt8),
        pl.col("kmers_lineage").cast(pl.UInt8), pl.col("kmers_misclassified").cast(pl.UInt8),
        pl.col("clade_ratio").cast(pl.Float32), pl.col("lineage_ratio").cast(pl.Float32),
        pl.col("misclassified_ratio").cast(pl.Float32),
    ])
    if output_path:
        metadata = get_standard_metadata(
            file_type="KRAKEN_PARQUET", source_path=output_path, 
            compression="Uncompressed", sorting="t_id", data_standard="SWEBITS"
        )
        write_parquet_with_metadata(df, output_path, metadata)
    return df

def generate_mock_report_parquet(
    sample_ids: List[str], 
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """Generates mock <REPORT_PARQUET> data."""
    data = []
    taxids = [9606, 10090, 5000000, 5000001, 5000002]
    for sid in sample_ids:
        info = parse_sample_id(sid)
        for tid in taxids:
            data.append({
                "sample_id": sid, "year": info["year"], "week": info["week"], "t_id": tid,
                "clade_reads": random.randint(100, 1000), "taxon_reads": random.randint(10, 100),
                "mm_tot": random.randint(1000, 5000), "mm_uniq": random.randint(500, 2000),
            })
    df = pl.DataFrame(data).with_columns([
        pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32),
        pl.col("clade_reads").cast(pl.UInt32), pl.col("taxon_reads").cast(pl.UInt32),
        pl.col("mm_tot").cast(pl.UInt64), pl.col("mm_uniq").cast(pl.UInt32),
    ])
    if output_path:
        metadata = get_standard_metadata(
            file_type="REPORT_PARQUET", source_path=Path.cwd(), 
            compression="Uncompressed", sorting="year, week, sample_id, t_id", data_standard="SWEBITS"
        )
        write_parquet_with_metadata(df, output_path, metadata)
    return df

def generate_mock_taxonomy(output_dir: Path):
    """Generates a minimal NCBITaxonomy-style names.dmp and nodes.dmp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = [
        "1\t|\t1\t|\tno rank\t|", "2\t|\t1\t|\tsuperkingdom\t|", "2759\t|\t1\t|\tsuperkingdom\t|",
        "9606\t|\t2759\t|\tspecies\t|", "10090\t|\t2759\t|\tspecies\t|", "5000000\t|\t2\t|\tgenus\t|",
        "5000001\t|\t5000000\t|\tspecies\t|", "5000002\t|\t5000000\t|\tspecies\t|",
    ]
    names = [
        "1\t|\troot\t|\t\t|\tscientific name\t|", "2\t|\tBacteria\t|\t\t|\tscientific name\t|",
        "2759\t|\tEukaryota\t|\t\t|\tscientific name\t|", "9606\t|\tHomo sapiens\t|\t\t|\tscientific name\t|",
        "10090\t|\tMus musculus\t|\t\t|\tscientific name\t|", "5000000\t|\tMockGenus\t|\t\t|\tscientific name\t|",
        "5000001\t|\t5000000\t|\tspecies\t|", "5000002\t|\t5000000\t|\tspecies\t|",
    ]
    with open(output_dir / "nodes.dmp", "w") as f:
        for line in nodes: f.write(line + "\n")
    with open(output_dir / "names.dmp", "w") as f:
        for line in names: f.write(line + "\n")
