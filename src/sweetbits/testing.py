import polars as pl
import numpy as np
import random
from typing import Optional
from pathlib import Path
from sweetbits.utils import parse_sample_id

def generate_random_dna(length: int) -> str:
    return "".join(random.choice("ACGT") for _ in range(length))

def generate_random_qual(length: int) -> str:
    # ASCII 33 (!) to 126 (~) are quality scores
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
        
        # k-mers per read: L - k + 1
        r1_kmers = max(0, r1_len - k + 1)
        r2_kmers = max(0, r2_len - k + 1)
        kmers_total = r1_kmers + r2_kmers
        
        if kmers_total == 0:
            kmers_total = 1 # ensure at least 1 for ratios
            
        kmers_ambig = random.randint(0, min(5, kmers_total))
        kmers_clade = random.randint(0, kmers_total - kmers_ambig)
        kmers_lineage = random.randint(kmers_clade, kmers_total - kmers_ambig)
        kmers_misclassified = kmers_total - kmers_ambig - kmers_lineage
        
        data.append({
            "sample_id": sample_id,
            "year": sample_info["year"],
            "week": sample_info["week"],
            "read_id": f"read_{i}",
            "r1_qual": generate_random_qual(r1_len),
            "r2_qual": generate_random_qual(r2_len),
            "r1_seq": generate_random_dna(r1_len),
            "r2_seq": generate_random_dna(r2_len),
            "r1_len": r1_len,
            "r2_len": r2_len,
            "total_len": total_len,
            "t_id": random.choice([9606, 10090, 5000001, 5000002]), # Human, Mouse, Mock GTDB
            "mhg": random.randint(1, 10),
            "kmer_string": "MOCK_KMER_STRING",
            "kmers_total": kmers_total,
            "kmers_ambig": kmers_ambig,
            "kmers_clade": kmers_clade,
            "kmers_lineage": kmers_lineage,
            "kmers_misclassified": kmers_misclassified,
            "clade_ratio": float(kmers_clade / kmers_total),
            "lineage_ratio": float(kmers_lineage / kmers_total),
            "misclassified_ratio": float(kmers_misclassified / kmers_total),
        })
    
    df = pl.DataFrame(data).with_columns([
        pl.col("year").cast(pl.UInt16),
        pl.col("week").cast(pl.UInt8),
        pl.col("r1_len").cast(pl.UInt8),
        pl.col("r2_len").cast(pl.UInt8),
        pl.col("total_len").cast(pl.UInt16),
        pl.col("t_id").cast(pl.UInt32),
        pl.col("mhg").cast(pl.UInt8),
        pl.col("kmers_total").cast(pl.UInt8),
        pl.col("kmers_ambig").cast(pl.UInt8),
        pl.col("kmers_clade").cast(pl.UInt8),
        pl.col("kmers_lineage").cast(pl.UInt8),
        pl.col("kmers_misclassified").cast(pl.UInt8),
        pl.col("clade_ratio").cast(pl.Float32),
        pl.col("lineage_ratio").cast(pl.Float32),
        pl.col("misclassified_ratio").cast(pl.Float32),
    ])
    
    if output_path:
        df.write_parquet(output_path)
    
    return df

def generate_mock_report_parquet(
    sample_ids: list[str], 
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """Generates mock <REPORT_PARQUET> data."""
    data = []
    taxids = [9606, 10090, 5000001, 5000002]
    
    for sid in sample_ids:
        info = parse_sample_id(sid)
        for tid in taxids:
            data.append({
                "sample_id": sid,
                "year": info["year"],
                "week": info["week"],
                "t_id": tid,
                "clade_reads": random.randint(100, 1000),
                "taxon_reads": random.randint(10, 100),
                "mm_tot": random.randint(1000, 5000),
                "mm_uniq": random.randint(500, 2000),
            })
            
    df = pl.DataFrame(data).with_columns([
        pl.col("year").cast(pl.UInt16),
        pl.col("week").cast(pl.UInt8),
        pl.col("t_id").cast(pl.UInt32),
        pl.col("clade_reads").cast(pl.UInt32),
        pl.col("taxon_reads").cast(pl.UInt32),
        pl.col("mm_tot").cast(pl.UInt64),
        pl.col("mm_uniq").cast(pl.UInt32),
    ])
    
    if output_path:
        df.write_parquet(output_path)
        
    return df
