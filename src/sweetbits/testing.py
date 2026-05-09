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
from sweetbits.metadata import get_standard_metadata, save_companion_metadata

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
        t_id = random.choice([9606, 10090, 5000101, 5000102])
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
        df.write_parquet(output_path)
        save_companion_metadata(output_path, metadata)
    return df

def generate_mock_report_parquet(
    sample_ids: List[str], 
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """Generates mock <REPORT_PARQUET> data."""
    data = []
    taxids = [9606, 10090, 562, 5000101, 5000102]
    for sid in sample_ids:
        info = parse_sample_id(sid)
        for tid in taxids:
            data.append({
                "sample_id": sid, "year": info["year"], "week": info["week"], "t_id": tid,
                "taxon_reads": random.randint(10, 100),
                "mm_tot": random.randint(1000, 5000), "mm_uniq": random.randint(500, 2000),
            })
    df = pl.DataFrame(data).with_columns([
        pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32),
        pl.col("taxon_reads").cast(pl.UInt32),
        pl.col("mm_tot").cast(pl.UInt64), pl.col("mm_uniq").cast(pl.UInt32),
    ])
    if output_path:
        metadata = get_standard_metadata(
            file_type="REPORT_PARQUET", source_path=Path.cwd(), 
            compression="Uncompressed", sorting="year, week, sample_id, t_id", data_standard="SWEBITS"
        )
        df.write_parquet(output_path)
        save_companion_metadata(output_path, metadata)
    return df

def generate_mock_taxonomy(output_dir: Path):
    """Generates an expansive, complex NCBITaxonomy-style tree spanning the tree of life with 100+ species."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Format: TaxID, ParentID, Rank, Name
    tax_data = [
        (1, 1, "no rank", "root"),
        # Domain: Eukaryota
        (2759, 1, "superkingdom", "Eukaryota"),
        (33208, 2759, "kingdom", "Metazoa"),
        (7711, 33208, "phylum", "Chordata"),
        (40674, 7711, "class", "Mammalia"),
        (9443, 40674, "order", "Primates"),
        (9605, 9443, "genus", "Homo"),
        (9606, 9605, "species", "Homo sapiens"),
        (9989, 40674, "order", "Rodentia"),
        (10088, 9989, "family", "Muridae"),
        (10089, 10088, "genus", "Mus"),
        (10090, 10089, "species", "Mus musculus"),
        (10114, 10089, "species", "Mus spretus"),
        (33090, 2759, "kingdom", "Viridiplantae"),
        (3193, 33090, "phylum", "Streptophyta"),
        (3700, 3193, "order", "Brassicales"),
        (3701, 3700, "genus", "Arabidopsis"),
        (3702, 3701, "species", "Arabidopsis thaliana"),
        
        # Domain: Bacteria
        (2, 1, "superkingdom", "Bacteria"),
        (1224, 2, "phylum", "Proteobacteria"),
        (28211, 1224, "class", "Alphaproteobacteria"),
        (91347, 28211, "order", "Enterobacterales"),
        (543, 91347, "family", "Enterobacteriaceae"),
        (561, 543, "genus", "Escherichia"),
        (562, 561, "species", "Escherichia coli"),
        (511145, 562, "strain", "Escherichia coli str. K-12"), 
        (590, 543, "genus", "Salmonella"),
        (28901, 590, "species", "Salmonella enterica"),
        (1239, 2, "phylum", "Firmicutes"),
        (91061, 1239, "class", "Bacilli"),
        (1385, 91061, "order", "Bacillales"),
        (1279, 1385, "family", "Staphylococcaceae"),
        (1281, 1279, "genus", "Staphylococcus"), 
        (1280, 1281, "species", "Staphylococcus aureus"),
        
        # Domain: Archaea
        (2157, 1, "superkingdom", "Archaea"),
        (28890, 2157, "phylum", "Euryarchaeota"),
        (183925, 28890, "class", "Methanobacteria"),
        (2158, 183925, "order", "Methanobacteriales"),
        (2159, 2158, "family", "Methanobacteriaceae"),
        (2162, 2159, "genus", "Methanobrevibacter"),
        (2168, 2162, "species", "Methanobrevibacter smithii"),
        
        # Domain: Viruses
        (10239, 1, "superkingdom", "Viruses"),
        (11118, 10239, "family", "Coronaviridae"),
        (11119, 11118, "genus", "Betacoronavirus"),
        (694009, 11119, "species", "Severe acute respiratory syndrome-related coronavirus"),
    ]
    
    # --- Generate Mass Dummy Lineages ---
    # We'll create 10 phyla under Bacteria, each with 2 genera, each with 5 species.
    # Total 10 * 2 * 5 = 100 species.
    for p in range(10):
        p_tid = 6000000 + p
        tax_data.append((p_tid, 2, "phylum", f"DummyPhylum_{p}"))
        for g in range(2):
            g_tid = p_tid * 10 + g
            tax_data.append((g_tid, p_tid, "genus", f"DummyGenus_{p}_{g}"))
            for s in range(5):
                s_tid = g_tid * 10 + s
                tax_data.append((s_tid, g_tid, "species", f"DummySpecies_{p}_{g}_{s}"))

    nodes = []
    names = []
    for tid, pid, rank, name in tax_data:
        nodes.append(f"{tid}\t|\t{pid}\t|\t{rank}\t|")
        names.append(f"{tid}\t|\t{name}\t|\t\t|\tscientific name\t|")

    with open(output_dir / "nodes.dmp", "w") as f:
        for line in nodes: f.write(line + "\n")
    with open(output_dir / "names.dmp", "w") as f:
        for line in names: f.write(line + "\n")
