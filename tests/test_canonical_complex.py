"""
tests.test_canonical_complex
Stress tests for canonical remainder logic with complex taxonomic structures and remainder bubbling.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from sweetbits.tables import generate_table_logic
from sweetbits.metadata import save_companion_metadata, get_standard_metadata
from joltax import JolTree

@pytest.fixture
def complex_taxonomy(tmp_path):
    """Creates a deep and broad taxonomy with nested non-canonical ranks."""
    tax_dir = tmp_path / "complex_tax"
    tax_dir.mkdir()
    
    # Format: TID | ParentTID | Rank
    nodes = [
        "1\t|\t1\t|\tno rank\t|",
        "2\t|\t1\t|\tsuperkingdom\t|",
        "10\t|\t2\t|\tphylum\t|",
        "100\t|\t10\t|\tclass\t|",
        "1000\t|\t100\t|\tsubclass\t|",      # Non-canonical
        "10000\t|\t1000\t|\torder\t|",
        "100000\t|\t10000\t|\tfamily\t|",
        "1000000\t|\t100000\t|\tgenus\t|",    # Direct child of Family
        "10000000\t|\t1000000\t|\tspecies\t|",
        "10000001\t|\t10000000\t|\tsubspecies\t|", # Non-canonical child of Species
        "10000002\t|\t10000000\t|\tstrain\t|",     # Another non-canonical child
        
        "20\t|\t2\t|\tphylum\t|",
        "200\t|\t20\t|\tno rank\t|",           # Non-canonical parent
        "2000\t|\t200\t|\tgenus\t|",           # Child of non-canonical
        "3000\t|\t200\t|\tgenus\t|",           # Sibling Genus
        
        "4000\t|\t200\t|\tgenus\t|",
        "40000\t|\t4000\t|\tspecies\t|",
        "40001\t|\t4000\t|\tspecies\t|"
    ]
    
    names = [f"{n.split()[0]}\t|\tName_{n.split()[0]}\t|\t\t|\tscientific name\t|" for n in nodes]
    
    with open(tax_dir / "nodes.dmp", "w") as f:
        for l in nodes: f.write(l + "\n")
    with open(tax_dir / "names.dmp", "w") as f:
        for l in names: f.write(l + "\n")
        
    tree = JolTree(tax_dir=str(tax_dir))
    cache_dir = tmp_path / "complex_cache"
    tree.save(str(cache_dir))
    return cache_dir

def test_canonical_deep_nesting(complex_taxonomy, tmp_path):
    """Tests remainder calculation through deep non-canonical gaps."""
    # Full exhaustive hierarchy from Root to Species
    data = pl.DataFrame({
        "sample_id": ["S1"] * 9,
        "year": [2022] * 9, "week": [1] * 9,
        "t_id": [1, 2, 10, 100, 1000, 10000, 100000, 1000000, 10000000],
        "taxon_reads": [0, 50, 50, 50, 0, 50, 50, 50, 50], # Total reads in scope = 350
        "mm_tot": [0]*9, "mm_uniq": [0]*9, "source_file": ["f"]*9
    }).with_columns([pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32)])
    
    report_parquet = tmp_path / "deep.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    data.write_parquet(report_parquet)
    save_companion_metadata(report_parquet, meta)
    
    out = tmp_path / "out.tsv"
    generate_table_logic(report_parquet, out, mode="canonical", taxonomy_dir=complex_taxonomy, min_observed=0, min_reads=0)
    
    res = {row["t_id"]: row["2022_01"] for row in pl.read_csv(out, separator="\t").to_dicts()}
    
    assert res[10000000] == 50 
    assert res[1000000] == 50   
    assert res[100000] == 50    
    assert res[100] == 50
    assert res[10] == 50
    assert res[2] == 50 # Absorbs Root(0) + SK(50) = 50.
    assert sum(res.values()) == 350

def test_canonical_bubbling_failure(complex_taxonomy, tmp_path):
    """Tests if a node that fails the min_reads filter pushes its mass to its NCA."""
    # Genus 4000 has 10 reads, Species 40000 has 10 reads, Species 40001 has 10 reads
    data = pl.DataFrame({
        "sample_id": ["S1"] * 4,
        "year": [2022] * 4, "week": [1] * 4,
        "t_id": [20, 4000, 40000, 40001],
        "taxon_reads": [50, 10, 10, 10], 
        "mm_tot": [0]*4, "mm_uniq": [0]*4, "source_file": ["f"]*4
    }).with_columns([pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32)])
    
    report_parquet = tmp_path / "bubble.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    data.write_parquet(report_parquet)
    save_companion_metadata(report_parquet, meta)
    
    out = tmp_path / "out_bubble.tsv"
    # Setting min_reads to 20
    # Raw remainders:
    # Species 40000: 10
    # Species 40001: 10
    # Genus 4000: 10 (30 clade - 10 - 10)
    # Phylum 20: 50
    # Filter 20: Species 40000 (10) fails, pushes 10 to Genus 4000
    # Filter 20: Species 40001 (10) fails, pushes 10 to Genus 4000
    # Genus 4000 remainder is now 10 + 10 + 10 = 30.
    # Genus 4000 (30 >= 20) passes! 
    
    generate_table_logic(report_parquet, out, mode="canonical", taxonomy_dir=complex_taxonomy, min_observed=1, min_reads=20)
    
    res = {row["t_id"]: row["2022_01"] for row in pl.read_csv(out, separator="\t").to_dicts()}
    
    assert 40000 not in res
    assert 40001 not in res
    assert res[4000] == 30
    assert res[20] == 50
    assert sum(res.values()) == 80 # Perfect mass balance

def test_canonical_subspecies_absorption(complex_taxonomy, tmp_path):
    """Tests that leaf canonical ranks natively absorb non-canonical descendants."""
    # Species 10000000 = 10 reads
    # Subspecies 10000001 = 20 reads
    # Strain 10000002 = 30 reads
    # Total mass = 60
    data = pl.DataFrame({
        "sample_id": ["S1"] * 3,
        "year": [2022] * 3, "week": [1] * 3,
        "t_id": [10000000, 10000001, 10000002],
        "taxon_reads": [10, 20, 30], 
        "mm_tot": [0]*3, "mm_uniq": [0]*3, "source_file": ["f"]*3
    }).with_columns([pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32)])
    
    report_parquet = tmp_path / "subsp.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    data.write_parquet(report_parquet)
    save_companion_metadata(report_parquet, meta)
    
    out = tmp_path / "out_subsp.tsv"
    
    # We set filter to 50. If Species didn't absorb its children before evaluating, 
    # it would fail. But since raw remainder of Species = 10+20+30 = 60, it passes.
    generate_table_logic(report_parquet, out, mode="canonical", taxonomy_dir=complex_taxonomy, min_observed=1, min_reads=50)
    
    res = {row["t_id"]: row["2022_01"] for row in pl.read_csv(out, separator="\t").to_dicts()}
    
    assert 10000001 not in res
    assert 10000002 not in res
    assert res[10000000] == 60
    assert sum(res.values()) == 60
